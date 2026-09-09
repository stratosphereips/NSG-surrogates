"""Fit the policy to labelled (state, action) pairs by supervised learning.

Each of the four heads receives a cross-entropy loss against the index that
`factorization.head_targets` determines it should have produced, masked to the
candidates available at that step. Each head is conditioned on the correct
earlier choices rather than on its own (teacher forcing; see
`FactoredGNNPolicy.head_logits`), so no head is trained to compensate for an
earlier head's error.

Two properties of the procedure matter as much as the loss:

* **Samples are re-projected from the copied state graphs**, not read from
  `projection_cache`. The graph is the dataset's authoritative content, so a
  change to the projection changes the training data. The cached projection is
  used only for comparison, and a difference is reported.
* **Data is split by run, not by row.** Several rows can refer to the same
  state, so a row-level split would place the same state in both parts and the
  reported accuracy would not measure generalisation.

Two reference values are reported alongside accuracy, because accuracy alone is
not interpretable at this sample size: the accuracy of always predicting the
most frequent action type, and the highest accuracy any function of the
encoder's input could reach given the labels (see `deterministic_ceiling`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from netsecgame.game_components import Action, GameState

from . import candidates as candidates_mod
from . import factorization
from .attempt_counts import AttemptCounts
from .dataset import read_pairs
from .encoder import state_summary, state_to_pyg
from .factorization import HeadTargets
from .policy import FactoredGNNPolicy
from .state_adapter import AdapterConfig, Projection, project_graph_to_game_state

HEADS = ("type", "source", "target", "secondary")


@dataclass
class Sample:
    """One trainable pair: a projected state, its candidates, and the target."""

    run_id: str
    pair_id: str
    state_id: str
    state: GameState
    projection: Projection
    action: Action
    candidates: List[Action]
    targets: HeadTargets
    weight: float
    label_source: str
    action_type: str
    #: Interaction history as of this step, replayed in trajectory order.
    attempt_counts: Optional[AttemptCounts] = None
    order_key: Tuple[str, str] = ("", "")

    @property
    def state_key(self) -> Tuple[str, str]:
        return (self.run_id, self.state_id)


@dataclass
class LoadReport:
    rows: int = 0
    samples: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    projection_mismatches: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    runs: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "samples": self.samples,
            "skipped": self.skipped,
            "projection_mismatches": self.projection_mismatches,
            "by_type": self.by_type,
            "by_source": self.by_source,
            "runs": self.runs,
        }


def load_samples(
    dataset_dirs: Sequence[str],
    config: Optional[AdapterConfig] = None,
    weight_by_confidence: bool = True,
) -> Tuple[List[Sample], LoadReport]:
    """Read `pairs.jsonl` files and rebuild trainable samples from the graphs."""
    config = config or AdapterConfig()
    report = LoadReport()
    skipped: Counter = Counter()
    samples: List[Sample] = []
    projections: Dict[Tuple[str, str], Projection] = {}

    for dataset_dir in dataset_dirs:
        pairs_path = os.path.join(dataset_dir, "pairs.jsonl")
        if not os.path.isfile(pairs_path):
            raise FileNotFoundError(f"no pairs.jsonl in {dataset_dir}")

        for row in read_pairs(pairs_path):
            report.rows += 1
            run_id = str(row.get("run_id", os.path.basename(dataset_dir)))
            state_ref = row.get("state_before") or {}
            state_id = str(state_ref.get("id", "?"))

            key = (dataset_dir, state_id)
            projection = projections.get(key)
            if projection is None:
                graph_path = os.path.join(dataset_dir, str(state_ref.get("graph", "")))
                if not os.path.isfile(graph_path):
                    skipped["state graph missing"] += 1
                    continue
                with open(graph_path, "r", encoding="utf-8") as handle:
                    graph = json.load(handle)
                projection = project_graph_to_game_state(
                    graph, config=config, source=graph_path
                )
                projections[key] = projection

            cached = (row.get("projection_cache") or {}).get("game_state")
            if cached is not None and not _same_state(cached, projection.state):
                # The adapter has changed since the dataset was written. Not
                # fatal — the graph is the source of truth and has just been
                # re-projected — but the caller should know the rows are stale.
                report.projection_mismatches += 1

            try:
                action = Action.from_dict(row["label"]["action"])
            except (KeyError, ValueError) as error:
                skipped[f"unreadable label ({type(error).__name__})"] += 1
                continue

            actions = candidates_mod.enumerate_actions(projection.state)
            targets = factorization.head_targets(
                action, actions, _object_to_idx(projection)
            )
            if targets is None:
                # The label is not reachable through the factorization from this
                # state — training on it would optimise an index that decodes to
                # something else.
                skipped["label not reachable from the candidate set"] += 1
                continue

            confidence = float(row["label"].get("confidence", 1.0))
            samples.append(
                Sample(
                    order_key=(state_id, str(row.get("pair_id", ""))),
                    run_id=run_id,
                    pair_id=str(row.get("pair_id", "")),
                    state_id=state_id,
                    state=projection.state,
                    projection=projection,
                    action=action,
                    candidates=actions,
                    targets=targets,
                    weight=confidence if weight_by_confidence else 1.0,
                    label_source=str(row["label"].get("label_source", "?")),
                    action_type=action.type.name,
                )
            )

    samples = _replay_attempt_counts(samples)
    report.samples = len(samples)
    report.skipped = dict(skipped)
    report.by_type = dict(Counter(sample.action_type for sample in samples))
    report.by_source = dict(Counter(sample.label_source for sample in samples))
    report.runs = dict(Counter(sample.run_id for sample in samples))
    return samples, report


def _replay_attempt_counts(samples: List[Sample]) -> List[Sample]:
    """Attach each run's interaction history, in recorded order.

    Without it the encoder produces the same input for the first and the fifth
    `FindData` on a host, so identical inputs carry different labels and no
    function of the input can fit them. On the two real runs, 21 samples reduce
    to 8 distinct inputs and the highest achievable accuracy is 15/21. The
    attempt counters exist to distinguish these cases, so the loader replays
    each run in order and records the counts as of each step.
    """
    ordered: List[Sample] = []
    for run_id in sorted({sample.run_id for sample in samples}):
        counts = AttemptCounts()
        run_samples = sorted(
            (sample for sample in samples if sample.run_id == run_id),
            key=lambda sample: sample.order_key,
        )
        for sample in run_samples:
            sample.attempt_counts = counts.snapshot()
            counts.record(sample.action)
            ordered.append(sample)
    return ordered


def _same_state(cached: Dict[str, Any], state: GameState) -> bool:
    """Compare a cached projection to a fresh one by value, not by JSON text.

    `GameState.as_json` orders sets arbitrarily, so comparing the serialised
    forms reports differences that do not exist.
    """
    try:
        return GameState.from_json(json.dumps(cached)) == state
    except Exception:
        return False


def _object_to_idx(projection: Projection) -> Dict[str, dict]:
    _, object_to_idx, _ = state_to_pyg(
        projection.state, provenance=projection.provenance
    )
    return object_to_idx


def split_by_run(
    samples: Sequence[Sample], holdout_runs: Sequence[str] = ()
) -> Tuple[List[Sample], List[Sample]]:
    """Split samples into training and evaluation parts, by run.

    With no run named, every sample is training data and the evaluation is
    therefore on the training set. That is reported as such rather than
    approximated with a row-level split, which would place the same state in
    both parts.
    """
    if not holdout_runs:
        return list(samples), []
    holdout = set(holdout_runs)
    return (
        [sample for sample in samples if sample.run_id not in holdout],
        [sample for sample in samples if sample.run_id in holdout],
    )


def _encode(sample: Sample):
    graph, object_to_idx, _ = state_to_pyg(
        sample.state,
        attempt_counts=sample.attempt_counts,
        provenance=sample.projection.provenance,
    )
    summary = state_summary(sample.state, provenance=sample.projection.provenance)
    return graph, object_to_idx, summary


def head_losses(
    policy: FactoredGNNPolicy, sample: Sample
) -> Dict[str, torch.Tensor]:
    """Cross-entropy loss for each active head of one sample."""
    graph, object_to_idx, summary = _encode(sample)
    logits = policy.head_logits(
        graph, object_to_idx, sample.candidates, sample.targets, summary=summary
    )

    losses: Dict[str, torch.Tensor] = {}
    indices = {
        "type": sample.targets.type_index,
        "source": sample.targets.source_index,
        "target": sample.targets.target_index,
        "secondary": sample.targets.secondary_index,
    }
    for head, index in indices.items():
        if index is None or head not in logits:
            continue
        head_logit = logits[head]
        if head_logit.numel() <= 1:
            # A head with one candidate conveys no information: its output is
            # determined, so a loss on it cannot change any selection.
            continue
        losses[head] = F.cross_entropy(
            head_logit.unsqueeze(0), torch.tensor([index], device=head_logit.device)
        )
    return losses


@dataclass
class EvalResult:
    samples: int = 0
    action_accuracy: float = 0.0
    type_accuracy: float = 0.0
    head_accuracy: Dict[str, float] = field(default_factory=dict)
    majority_type_baseline: float = 0.0
    deterministic_ceiling: float = 1.0
    distinct_inputs: int = 0
    fallbacks: int = 0
    mean_candidates: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "action_accuracy": round(self.action_accuracy, 4),
            "type_accuracy": round(self.type_accuracy, 4),
            "head_accuracy": {k: round(v, 4) for k, v in self.head_accuracy.items()},
            "majority_type_baseline": round(self.majority_type_baseline, 4),
            "deterministic_ceiling": round(self.deterministic_ceiling, 4),
            "distinct_inputs": self.distinct_inputs,
            "fallbacks": self.fallbacks,
            "mean_candidates": round(self.mean_candidates, 2),
        }


@torch.no_grad()
def evaluate(policy: FactoredGNNPolicy, samples: Sequence[Sample]) -> EvalResult:
    """Select an action greedily for every sample and compare with its label."""
    result = EvalResult(samples=len(samples))
    if not samples:
        return result

    policy.eval()
    correct_action = 0
    correct_type = 0
    head_hits: Counter = Counter()
    head_total: Counter = Counter()

    for sample in samples:
        graph, object_to_idx, summary = _encode(sample)
        decision = policy.decide(
            graph, object_to_idx, sample.candidates, temperature=1.0, summary=summary, greedy=True
        )
        if decision.fallback:
            result.fallbacks += 1
            continue

        if candidates_mod.action_key(decision.action) == candidates_mod.action_key(
            sample.action
        ):
            correct_action += 1
        if decision.action.type == sample.action.type:
            correct_type += 1

        predicted = factorization.head_targets(
            decision.action, sample.candidates, object_to_idx
        )
        if predicted is None:
            continue
        for head, truth, guess in (
            ("type", sample.targets.type_index, predicted.type_index),
            ("source", sample.targets.source_index, predicted.source_index),
            ("target", sample.targets.target_index, predicted.target_index),
            ("secondary", sample.targets.secondary_index, predicted.secondary_index),
        ):
            if truth is None:
                continue
            head_total[head] += 1
            if truth == guess:
                head_hits[head] += 1

    result.action_accuracy = correct_action / len(samples)
    result.type_accuracy = correct_type / len(samples)
    result.head_accuracy = {
        head: head_hits[head] / head_total[head] for head in HEADS if head_total[head]
    }
    counts = Counter(sample.action_type for sample in samples)
    result.majority_type_baseline = max(counts.values()) / len(samples)
    ceiling, distinct = deterministic_ceiling(samples)
    result.deterministic_ceiling = ceiling
    result.distinct_inputs = distinct
    result.mean_candidates = sum(len(s.candidates) for s in samples) / len(samples)
    return result


def deterministic_ceiling(samples: Sequence[Sample]) -> Tuple[float, int]:
    """The highest accuracy any function of the encoder's input can reach.

    Samples are grouped by what the encoder receives — the projected state and
    the attempt counts — and only the most frequent label in each group is
    counted as reachable. Identical inputs with different labels cannot both be
    satisfied, so accuracy equal to this value means the remaining errors are
    contradictory labels rather than underfitting.
    """
    if not samples:
        return 1.0, 0
    groups: Dict[str, Counter] = {}
    for sample in samples:
        groups.setdefault(_input_key(sample), Counter())[
            candidates_mod.action_sort_key(sample.action)
        ] += 1
    reachable = sum(max(counter.values()) for counter in groups.values())
    return reachable / len(samples), len(groups)


def _input_key(sample: Sample) -> str:
    """The inputs the encoder derives its features from, as a hashable key."""
    counts = sample.attempt_counts
    history = (
        json.dumps(
            {
                name: sorted((str(key), value) for key, value in mapping.items())
                for name, mapping in (
                    ("scan", counts.scan),
                    ("findservices", counts.findservices),
                    ("finddata", counts.finddata),
                    ("exploit", counts.exploit),
                    ("exfil", counts.exfil),
                )
            },
            sort_keys=True,
        )
        if counts is not None
        else ""
    )
    return sample.state.as_json() + "|" + history


@dataclass
class TrainResult:
    load: LoadReport
    epochs: List[Dict[str, Any]] = field(default_factory=list)
    train_eval: EvalResult = field(default_factory=EvalResult)
    holdout_eval: EvalResult = field(default_factory=EvalResult)
    weights_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "load": self.load.as_dict(),
            "epochs": self.epochs,
            "train_eval": self.train_eval.as_dict(),
            "holdout_eval": self.holdout_eval.as_dict(),
            "weights_path": self.weights_path,
        }


def train(
    samples: Sequence[Sample],
    holdout: Sequence[Sample] = (),
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    policy: Optional[FactoredGNNPolicy] = None,
    load: Optional[LoadReport] = None,
    verbose: bool = True,
) -> TrainResult:
    """Fit the four heads by masked cross-entropy.

    Samples are processed individually. Each state has a different graph and a
    different number of candidates, so batching would require padding machinery
    that this quantity of data does not justify.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    policy = policy or FactoredGNNPolicy()
    result = TrainResult(load=load or LoadReport(samples=len(samples)))
    if not samples:
        return result

    optimizer = torch.optim.Adam(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    order = list(range(len(samples)))

    for epoch in range(1, epochs + 1):
        policy.train()
        random.shuffle(order)
        totals: Counter = Counter()
        counts: Counter = Counter()
        epoch_loss = 0.0

        for index in order:
            sample = samples[index]
            losses = head_losses(policy, sample)
            if not losses:
                continue
            loss = sample.weight * torch.stack(list(losses.values())).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            for head, value in losses.items():
                totals[head] += float(value.item())
                counts[head] += 1

        record = {
            "epoch": epoch,
            "loss": round(epoch_loss / max(len(order), 1), 4),
            "per_head": {
                head: round(totals[head] / counts[head], 4) for head in HEADS if counts[head]
            },
        }
        if epoch == epochs or epoch % max(epochs // 10, 1) == 0:
            record["train_action_accuracy"] = round(evaluate(policy, samples).action_accuracy, 4)
            if holdout:
                record["holdout_action_accuracy"] = round(
                    evaluate(policy, holdout).action_accuracy, 4
                )
        result.epochs.append(record)
        if verbose:
            print(f"epoch {epoch:>3}  loss {record['loss']:<9} {record.get('per_head', {})}")

    result.train_eval = evaluate(policy, samples)
    result.holdout_eval = evaluate(policy, holdout)
    return result


def save(policy: FactoredGNNPolicy, path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(policy.state_dict(), path)
    return path
