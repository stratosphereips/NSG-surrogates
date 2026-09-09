"""The policy: graph tensors in, one parameterised NetSecGame action out.

The architecture is taken from `sgrl_netsec/blackbox_pure_gnn_agent.py`: a
network body shared by four heads that select an action type, a source host, a
primary target and, for exfiltration, a destination. That is provenance rather
than a constraint. Parameter compatibility with that agent is not a
requirement, so the two may diverge; anything gained by changing the feature
layout or the head structure is worth more than the ability to exchange
checkpoints.

`SurrogatePolicy` adds what is needed outside the simulator: per-episode attempt
counters, the projection's `Provenance`, and a `Decision` record of what each
head selected, so that a chosen action can be explained after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from netsecgame.game_components import Action, ActionType, GameState

from . import candidates as candidates_mod
from . import factorization
from .action_schema import (
    ACTION_SECONDARY_NODE_TYPE,
    ACTION_SECONDARY_PARAM,
    ACTION_TARGET_NODE_TYPE,
    ACTION_TARGET_PARAM,
    ACTION_TYPE_LIST,
    SECONDARY_HEAD_ACTION_TYPES,
)
from .attempt_counts import AttemptCounts
from .encoder import EDGE_TYPES, FEATURE_DIM, NODE_TYPES, state_summary, state_to_pyg
from .factorization import HeadTargets
from .state_adapter import Provenance


@dataclass
class Decision:
    """One selected action, with the choice each of the four heads made."""

    action: Action
    head_choices: Dict[str, str] = field(default_factory=dict)
    log_prob: Optional[float] = None
    entropy: Optional[float] = None
    candidate_count: int = 0
    fallback: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.as_dict,
            "action_human": candidates_mod.describe(self.action),
            "head_choices": self.head_choices,
            "log_prob": self.log_prob,
            "entropy": self.entropy,
            "candidate_count": self.candidate_count,
            "fallback": self.fallback,
        }


class FactoredGNNPolicy(nn.Module):
    """A GATv2 network body with four sequential, candidate-masked heads.

      type_head : pooled graph ‖ summary           -> action type (5 classes)
      src_head  : host ‖ pooled ‖ type             -> source host
      tgt_head  : node ‖ pooled ‖ type ‖ src       -> primary target
      sec_head  : node ‖ pooled ‖ type ‖ src ‖ tgt -> secondary parameter

    Each head is masked to the candidates available at that step, so every
    selected action is valid by construction. The heads run in sequence, each
    conditioned on the previous choices, so the joint log-probability of an
    action is the sum of the four terms.
    """

    NUM_STATE_SUMMARY = 3

    def __init__(self, hidden_channels: int = 64, num_gnn_layers: int = 2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, HeteroConv, Linear

        H = hidden_channels
        self.hidden_channels = H
        num_types = len(ACTION_TYPE_LIST)

        self.node_encoders = nn.ModuleDict(
            {node_type: Linear(FEATURE_DIM, H) for node_type in NODE_TYPES}
        )
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {
                        edge_type: GATv2Conv((-1, -1), H, add_self_loops=False)
                        for edge_type in EDGE_TYPES
                    },
                    aggr="sum",
                )
                for _ in range(num_gnn_layers)
            ]
        )
        self.type_emb = nn.Embedding(num_types, H)
        self.type_head = nn.Sequential(
            nn.Linear(H + self.NUM_STATE_SUMMARY, H), nn.ReLU(), nn.Linear(H, num_types)
        )
        self.src_head = nn.Sequential(nn.Linear(H * 3, H), nn.ReLU(), nn.Linear(H, 1))
        self.tgt_head = nn.Sequential(nn.Linear(H * 4, H), nn.ReLU(), nn.Linear(H, 1))
        self.sec_head = nn.Sequential(nn.Linear(H * 5, H), nn.ReLU(), nn.Linear(H, 1))
        self.value_head = nn.Sequential(
            nn.Linear(H + self.NUM_STATE_SUMMARY, H), nn.ReLU(), nn.Linear(H, 1)
        )

    # ── backbone ─────────────────────────────────────────────────────────────

    def _gnn_forward(self, data) -> Dict[str, torch.Tensor]:
        embeddings = {}
        for node_type in NODE_TYPES:
            if node_type in data.x_dict:
                embeddings[node_type] = F.relu(self.node_encoders[node_type](data.x_dict[node_type]))
        for conv in self.convs:
            active = {key: value for key, value in data.edge_index_dict.items() if value.numel() > 0}
            if active:
                embeddings = conv(embeddings, active)
            embeddings = {node_type: F.relu(value) for node_type, value in embeddings.items()}
        return embeddings

    def _global_emb(self, embeddings: Dict[str, torch.Tensor], device) -> torch.Tensor:
        parts = [value.mean(dim=0) for value in embeddings.values() if value.numel() > 0]
        return (
            torch.stack(parts).mean(dim=0)
            if parts
            else torch.zeros(self.hidden_channels, device=device)
        )

    def state_value(self, data, summary: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Value estimate for a state, from the shared network body."""
        device = next(self.parameters()).device
        pooled = self._global_emb(self._gnn_forward(data), device)
        if summary is None:
            summary = torch.zeros(self.NUM_STATE_SUMMARY, device=device)
        return self.value_head(torch.cat([pooled, summary])).squeeze(-1)

    # ── decision ─────────────────────────────────────────────────────────────

    def head_logits(
        self,
        data,
        object_to_idx: Dict[str, dict],
        valid_actions: List[Action],
        targets: HeadTargets,
        summary: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Logits for every active head, conditioned on the correct prefix.

        This is teacher forcing. When selecting an action, each head is
        conditioned on what the previous heads chose; during supervised training
        it is conditioned on what they should have chosen, so that a later head
        is not trained to compensate for an earlier head's error.

        Candidate lists come from `factorization`, which action selection also
        uses, so index `i` denotes the same object in both directions.
        """
        device = next(self.parameters()).device
        zero = torch.zeros(self.hidden_channels, device=device)
        if summary is None:
            summary = torch.zeros(self.NUM_STATE_SUMMARY, device=device)

        embeddings = self._gnn_forward(data)
        pooled = self._global_emb(embeddings, device)
        logits: Dict[str, torch.Tensor] = {}

        mask = torch.tensor(
            factorization.type_mask(valid_actions), dtype=torch.bool, device=device
        )
        logits["type"] = self.type_head(torch.cat([pooled, summary])).masked_fill(~mask, -1e9)

        action_type = ACTION_TYPE_LIST[targets.type_index]
        type_context = self.type_emb(torch.tensor(targets.type_index, device=device))
        remaining = [a for a in valid_actions if a.type == action_type]

        source_embedding = zero
        sources = factorization.source_candidates(remaining, object_to_idx)
        if sources and targets.source_index is not None:
            host_embeddings = embeddings.get(
                "host", torch.empty(0, self.hidden_channels, device=device)
            )
            logits["source"] = torch.stack(
                [
                    self.src_head(torch.cat([host_embeddings[idx], pooled, type_context])).squeeze(-1)
                    for _, idx in sources
                ]
            )
            chosen_source, chosen_idx = sources[targets.source_index]
            source_embedding = host_embeddings[chosen_idx]
            remaining = factorization.narrow(remaining, "source_host", chosen_source)

        picks, target_param, target_node_type = factorization.target_candidates(
            remaining, action_type, object_to_idx
        )
        target_embedding = zero
        if picks and targets.target_index is not None and target_node_type:
            node_embeddings = embeddings.get(
                target_node_type, torch.empty(0, self.hidden_channels, device=device)
            )
            logits["target"] = torch.stack(
                [
                    self.tgt_head(
                        torch.cat([node_embeddings[idx], pooled, type_context, source_embedding])
                    ).squeeze(-1)
                    for _, idx in picks
                ]
            )
            chosen_target, chosen_idx = picks[targets.target_index]
            target_embedding = node_embeddings[chosen_idx]
            remaining = factorization.narrow(remaining, target_param, chosen_target)

        secondaries, _, secondary_node_type = factorization.secondary_candidates(
            remaining, action_type, object_to_idx
        )
        if secondaries and targets.secondary_index is not None and secondary_node_type:
            node_embeddings = embeddings.get(
                secondary_node_type, torch.empty(0, self.hidden_channels, device=device)
            )
            logits["secondary"] = torch.stack(
                [
                    self.sec_head(
                        torch.cat(
                            [
                                node_embeddings[idx],
                                pooled,
                                type_context,
                                source_embedding,
                                target_embedding,
                            ]
                        )
                    ).squeeze(-1)
                    for _, idx in secondaries
                ]
            )

        return logits

    def decide(
        self,
        data,
        object_to_idx: Dict[str, dict],
        valid_actions: List[Action],
        temperature: float = 1.0,
        summary: Optional[torch.Tensor] = None,
        greedy: bool = False,
    ) -> Decision:
        """Select one action, recording each head's choice.

        With `greedy`, each head takes its argmax instead of sampling. This is
        what evaluation requires: accuracy against a labelled action is not
        defined if the same state yields a different action on each call.
        """
        device = next(self.parameters()).device
        zero = torch.zeros(self.hidden_channels, device=device)
        temperature = max(temperature, 1e-8)
        choices: Dict[str, str] = {}

        embeddings = self._gnn_forward(data)
        pooled = self._global_emb(embeddings, device)
        if summary is None:
            summary = torch.zeros(self.NUM_STATE_SUMMARY, device=device)

        def pick(distribution: Categorical, logits: torch.Tensor) -> torch.Tensor:
            return logits.argmax() if greedy else distribution.sample()

        def fallback(reason: str) -> Decision:
            # The factorisation cannot express this candidate set. Select
            # uniformly, report the reason, and return no log-probability, so
            # that the choice is not attributed to the heads.
            choices["fallback_reason"] = reason
            return Decision(
                action=random.choice(valid_actions),
                head_choices=choices,
                candidate_count=len(valid_actions),
                fallback=True,
            )

        # ── 1. action type ───────────────────────────────────────────────────
        type_mask = torch.tensor(
            factorization.type_mask(valid_actions), dtype=torch.bool, device=device
        )
        type_logits = self.type_head(torch.cat([pooled, summary])).masked_fill(~type_mask, -1e9)
        type_dist = Categorical(logits=type_logits / temperature)
        type_index = pick(type_dist, type_logits)
        chosen_type = ACTION_TYPE_LIST[int(type_index.item())]
        type_context = self.type_emb(type_index)
        log_prob = type_dist.log_prob(type_index)
        entropy = type_dist.entropy()
        choices["type"] = chosen_type.name

        remaining = [action for action in valid_actions if action.type == chosen_type]

        # ── 2. source host ───────────────────────────────────────────────────
        source_embedding = zero
        sources = factorization.source_candidates(remaining, object_to_idx)
        if sources:
            host_embeddings = embeddings.get(
                "host", torch.empty(0, self.hidden_channels, device=device)
            )
            scores = torch.stack(
                [
                    self.src_head(torch.cat([host_embeddings[idx], pooled, type_context])).squeeze(-1)
                    for _, idx in sources
                ]
            )
            source_dist = Categorical(logits=scores / temperature)
            picked = pick(source_dist, scores)
            chosen_source, source_index = sources[int(picked.item())]
            source_embedding = host_embeddings[source_index]
            log_prob = log_prob + source_dist.log_prob(picked)
            entropy = entropy + source_dist.entropy()
            choices["source_host"] = str(chosen_source)
            remaining = factorization.narrow(remaining, "source_host", chosen_source)

        # ── 3. primary target ────────────────────────────────────────────────
        picks, target_param, target_node_type = factorization.target_candidates(
            remaining, chosen_type, object_to_idx
        )
        if target_param is None or target_node_type is None:
            return fallback(f"no target schema for {chosen_type.name}")
        if not picks:
            return fallback(f"no {target_node_type} node for any valid {target_param}")

        target_embeddings = embeddings.get(
            target_node_type, torch.empty(0, self.hidden_channels, device=device)
        )
        scores = torch.stack(
            [
                self.tgt_head(
                    torch.cat([target_embeddings[idx], pooled, type_context, source_embedding])
                ).squeeze(-1)
                for _, idx in picks
            ]
        )
        target_dist = Categorical(logits=scores / temperature)
        picked = pick(target_dist, scores)
        chosen_target, target_index = picks[int(picked.item())]
        target_embedding = target_embeddings[target_index]
        log_prob = log_prob + target_dist.log_prob(picked)
        entropy = entropy + target_dist.entropy()
        choices[target_param] = str(chosen_target)
        remaining = factorization.narrow(remaining, target_param, chosen_target)

        # ── 4. secondary parameter ───────────────────────────────────────────
        secondaries, secondary_param, secondary_node_type = factorization.secondary_candidates(
            remaining, chosen_type, object_to_idx
        )
        if secondaries and secondary_param and secondary_node_type:
            secondary_embeddings = embeddings.get(
                secondary_node_type, torch.empty(0, self.hidden_channels, device=device)
            )
            scores = torch.stack(
                [
                    self.sec_head(
                        torch.cat(
                            [
                                secondary_embeddings[idx],
                                pooled,
                                type_context,
                                source_embedding,
                                target_embedding,
                            ]
                        )
                    ).squeeze(-1)
                    for _, idx in secondaries
                ]
            )
            secondary_dist = Categorical(logits=scores / temperature)
            picked = pick(secondary_dist, scores)
            chosen_secondary = secondaries[int(picked.item())][0]
            log_prob = log_prob + secondary_dist.log_prob(picked)
            entropy = entropy + secondary_dist.entropy()
            choices[f"secondary:{secondary_param}"] = str(chosen_secondary)
            remaining = factorization.narrow(remaining, secondary_param, chosen_secondary)

        if len(remaining) != 1:
            return fallback(f"{len(remaining)} actions still match after all heads")

        return Decision(
            action=remaining[0],
            head_choices=choices,
            log_prob=float(log_prob.item()),
            entropy=float(entropy.item()),
            candidate_count=len(valid_actions),
        )

    def forward(
        self,
        data,
        object_to_idx: Dict[str, dict],
        valid_actions: List[Action],
        temperature: float = 1.0,
        state_summary: Optional[torch.Tensor] = None,
    ) -> Tuple[Action, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Same call signature as the simulator agent's policy."""
        decision = self.decide(
            data, object_to_idx, valid_actions, temperature=temperature, summary=state_summary
        )
        if decision.fallback:
            return decision.action, None, None
        device = next(self.parameters()).device
        return (
            decision.action,
            torch.tensor(decision.log_prob, device=device),
            torch.tensor(decision.entropy, device=device),
        )


class SurrogatePolicy:
    """Holds the attempt counters and performs the encoding for each call."""

    def __init__(
        self,
        policy: Optional[FactoredGNNPolicy] = None,
        device: Optional[torch.device] = None,
        legacy_service_port: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = (policy or FactoredGNNPolicy()).to(self.device)
        self.policy.eval()
        self.attempt_counts = AttemptCounts()
        self.legacy_service_port = legacy_service_port

    @classmethod
    def load(
        cls, weights: Optional[str] = None, legacy_service_port: bool = False
    ) -> "SurrogatePolicy":
        """Load a checkpoint, or return a randomly initialised policy.

        A randomly initialised policy is a useful control: it exercises the
        whole path from state to action, and its action distribution is the
        baseline a fitted policy is compared against.
        """
        surrogate = cls(legacy_service_port=legacy_service_port)
        if weights:
            if not os.path.exists(weights):
                raise FileNotFoundError(f"weights not found: {weights}")
            state_dict = torch.load(weights, map_location=surrogate.device, weights_only=False)
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()
            surrogate.policy.load_state_dict(state_dict)
        return surrogate

    def reset(self) -> None:
        self.attempt_counts.reset()

    def act(
        self,
        state: GameState,
        valid_actions: Optional[List[Action]] = None,
        provenance: Optional[Provenance] = None,
        temperature: float = 0.1,
        record: bool = True,
        greedy: bool = False,
    ) -> Decision:
        actions = (
            list(valid_actions)
            if valid_actions is not None
            else candidates_mod.enumerate_actions(state)
        )
        if not actions:
            raise ValueError("no valid action for this state")

        graph, object_to_idx, _ = state_to_pyg(
            state,
            attempt_counts=self.attempt_counts,
            provenance=provenance,
            legacy_service_port=self.legacy_service_port,
        )
        summary = state_summary(state, provenance=provenance).to(self.device)
        with torch.no_grad():
            decision = self.policy.decide(
                graph.to(self.device),
                object_to_idx,
                actions,
                temperature=temperature,
                summary=summary,
                greedy=greedy,
            )
        if record:
            self.attempt_counts.record(decision.action)
        return decision
