"""The surrogate policy: hetero graph in, one parameterized NSG action out.

`FactoredGNNPolicy` is architecturally identical to
`sgrl_netsec/blackbox_pure_gnn_agent.py:FactoredGNNPolicy` — same module names,
shapes and head inputs — so `best_blackbox_gnn.pth` loads into it unchanged.
That is deliberate: the PoC's question is whether a policy trained in the
simulator can act on real container state, which is only answerable if the
weights transfer without surgery.

`SurrogatePolicy` wraps it with the pieces the real range needs: episode-local
attempt counters, the state adapter's provenance, and a `Decision` that records
what each head chose so an emitted action can be explained.
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
from .state_adapter import Provenance


@dataclass
class Decision:
    """One action plus the trace of how the four heads reached it."""

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
            "translator_support": candidates_mod.support_status(self.action),
            "head_choices": self.head_choices,
            "log_prob": self.log_prob,
            "entropy": self.entropy,
            "candidate_count": self.candidate_count,
            "fallback": self.fallback,
        }


class FactoredGNNPolicy(nn.Module):
    """GATv2 backbone plus four sequential, candidate-masked decision heads.

      type_head : pooled graph ‖ summary          -> action type   (5-way)
      src_head  : host ‖ pooled ‖ type            -> source host
      tgt_head  : node ‖ pooled ‖ type ‖ src      -> primary target
      sec_head  : node ‖ pooled ‖ type ‖ src ‖ tgt-> secondary (ExfiltrateData)

    Each head is masked to the candidate set, so every sampled action is legal
    by construction. Because the heads run in sequence, each conditioned on the
    previous choices, the joint log-probability is the exact sum of the four.
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
        device = next(self.parameters()).device
        pooled = self._global_emb(self._gnn_forward(data), device)
        if summary is None:
            summary = torch.zeros(self.NUM_STATE_SUMMARY, device=device)
        return self.value_head(torch.cat([pooled, summary])).squeeze(-1)

    # ── decision ─────────────────────────────────────────────────────────────

    def decide(
        self,
        data,
        object_to_idx: Dict[str, dict],
        valid_actions: List[Action],
        temperature: float = 1.0,
        summary: Optional[torch.Tensor] = None,
    ) -> Decision:
        """Sample one action, recording each head's choice."""
        device = next(self.parameters()).device
        zero = torch.zeros(self.hidden_channels, device=device)
        temperature = max(temperature, 1e-8)
        choices: Dict[str, str] = {}

        embeddings = self._gnn_forward(data)
        pooled = self._global_emb(embeddings, device)
        if summary is None:
            summary = torch.zeros(self.NUM_STATE_SUMMARY, device=device)

        def fallback(reason: str) -> Decision:
            # The factorization cannot express the candidate set. Act, but emit
            # no gradient-bearing terms and say so, rather than silently
            # pretending the heads made this choice.
            choices["fallback_reason"] = reason
            return Decision(
                action=random.choice(valid_actions),
                head_choices=choices,
                candidate_count=len(valid_actions),
                fallback=True,
            )

        # ── 1. action type ───────────────────────────────────────────────────
        valid_types = {action.type for action in valid_actions}
        type_mask = torch.tensor(
            [action_type in valid_types for action_type in ACTION_TYPE_LIST],
            dtype=torch.bool,
            device=device,
        )
        type_logits = self.type_head(torch.cat([pooled, summary])).masked_fill(~type_mask, -1e9)
        type_dist = Categorical(logits=type_logits / temperature)
        type_index = type_dist.sample()
        chosen_type = ACTION_TYPE_LIST[int(type_index.item())]
        type_context = self.type_emb(type_index)
        log_prob = type_dist.log_prob(type_index)
        entropy = type_dist.entropy()
        choices["type"] = chosen_type.name

        remaining = [action for action in valid_actions if action.type == chosen_type]

        # ── 2. source host ───────────────────────────────────────────────────
        source_embedding = zero
        valid_sources = {
            action.parameters["source_host"]
            for action in remaining
            if "source_host" in action.parameters
        }
        if valid_sources:
            host_embeddings = embeddings.get(
                "host", torch.empty(0, self.hidden_channels, device=device)
            )
            scores, keys = [], []
            for obj, idx in object_to_idx.get("host", {}).items():
                if obj in valid_sources:
                    scores.append(
                        self.src_head(torch.cat([host_embeddings[idx], pooled, type_context])).squeeze(-1)
                    )
                    keys.append((obj, idx))
            if scores:
                source_dist = Categorical(logits=torch.stack(scores) / temperature)
                pick = source_dist.sample()
                chosen_source, source_index = keys[int(pick.item())]
                source_embedding = host_embeddings[source_index]
                log_prob = log_prob + source_dist.log_prob(pick)
                entropy = entropy + source_dist.entropy()
                choices["source_host"] = str(chosen_source)
                remaining = [
                    action
                    for action in remaining
                    if action.parameters.get("source_host") == chosen_source
                ]

        # ── 3. primary target ────────────────────────────────────────────────
        target_param = ACTION_TARGET_PARAM.get(chosen_type)
        target_node_type = ACTION_TARGET_NODE_TYPE.get(chosen_type)
        if target_param is None or target_node_type is None:
            return fallback(f"no target schema for {chosen_type.name}")

        valid_targets = {
            action.parameters[target_param]
            for action in remaining
            if target_param in action.parameters
        }
        target_embeddings = embeddings.get(
            target_node_type, torch.empty(0, self.hidden_channels, device=device)
        )
        scores, keys = [], []
        for obj, idx in object_to_idx.get(target_node_type, {}).items():
            if obj in valid_targets:
                scores.append(
                    self.tgt_head(
                        torch.cat([target_embeddings[idx], pooled, type_context, source_embedding])
                    ).squeeze(-1)
                )
                keys.append((obj, idx))
        if not scores:
            return fallback(f"no {target_node_type} node for any valid {target_param}")

        target_dist = Categorical(logits=torch.stack(scores) / temperature)
        pick = target_dist.sample()
        chosen_target, target_index = keys[int(pick.item())]
        target_embedding = target_embeddings[target_index]
        log_prob = log_prob + target_dist.log_prob(pick)
        entropy = entropy + target_dist.entropy()
        choices[target_param] = str(chosen_target)
        remaining = [
            action for action in remaining if action.parameters.get(target_param) == chosen_target
        ]

        # ── 4. secondary parameter ───────────────────────────────────────────
        if chosen_type in SECONDARY_HEAD_ACTION_TYPES:
            secondary_param = ACTION_SECONDARY_PARAM.get(chosen_type)
            secondary_node_type = ACTION_SECONDARY_NODE_TYPE.get(chosen_type)
            if secondary_param and secondary_node_type:
                valid_secondaries = {
                    action.parameters[secondary_param]
                    for action in remaining
                    if secondary_param in action.parameters
                }
                secondary_embeddings = embeddings.get(
                    secondary_node_type, torch.empty(0, self.hidden_channels, device=device)
                )
                scores, keys = [], []
                for obj, idx in object_to_idx.get(secondary_node_type, {}).items():
                    if obj in valid_secondaries:
                        scores.append(
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
                        )
                        keys.append(obj)
                if scores:
                    secondary_dist = Categorical(logits=torch.stack(scores) / temperature)
                    pick = secondary_dist.sample()
                    chosen_secondary = keys[int(pick.item())]
                    log_prob = log_prob + secondary_dist.log_prob(pick)
                    entropy = entropy + secondary_dist.entropy()
                    choices[f"secondary:{secondary_param}"] = str(chosen_secondary)
                    remaining = [
                        action
                        for action in remaining
                        if action.parameters.get(secondary_param) == chosen_secondary
                    ]

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
        """Signature-compatible with the simulator agent's policy call."""
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
    """Stateful wrapper: keeps attempt counters and does the encoding."""

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

        A random policy is useful on purpose: it exercises the whole
        state -> graph -> action path without claiming the choices are good.
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
            )
        if record:
            self.attempt_counts.record(decision.action)
        return decision
