"""Per-head candidate lists, shared by action selection and target construction.

Two operations need the same lists:

* selecting an action narrows the candidate set head by head and samples an
  index from each;
* constructing a training target converts a labelled action back into the index
  each head should have produced.

If the two enumerations differed in iteration order or in filtering, training
would optimise indices that correspond to different actions at selection time,
and the loss would give no indication of it. The lists are therefore built once,
here, and used by both.

Node iteration follows `object_to_idx`, whose insertion order is the sorted
order that `state_to_pyg` produces, so candidate indices are stable across
calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from netsecgame.game_components import Action, ActionType

from .action_schema import (
    ACTION_SECONDARY_NODE_TYPE,
    ACTION_SECONDARY_PARAM,
    ACTION_TARGET_NODE_TYPE,
    ACTION_TARGET_PARAM,
    ACTION_TYPE_LIST,
    SECONDARY_HEAD_ACTION_TYPES,
)

#: One candidate for a node-scoring head: the game object and its node index.
NodeCandidate = Tuple[Any, int]


@dataclass(frozen=True)
class HeadTargets:
    """The index each head should emit to produce a given action.

    `None` means the head does not run for this action type (`secondary` for
    everything but `ExfiltrateData`), or the step was degenerate — a head with a
    single candidate still runs, but one with none cannot.
    """

    type_index: int
    source_index: Optional[int]
    target_index: Optional[int]
    secondary_index: Optional[int]

    @property
    def active_heads(self) -> Tuple[str, ...]:
        heads = ["type"]
        if self.source_index is not None:
            heads.append("source")
        if self.target_index is not None:
            heads.append("target")
        if self.secondary_index is not None:
            heads.append("secondary")
        return tuple(heads)


def type_mask(candidates: Sequence[Action]) -> List[bool]:
    """Which action types appear in the candidate set."""
    present = {action.type for action in candidates}
    return [action_type in present for action_type in ACTION_TYPE_LIST]


def source_candidates(
    candidates: Sequence[Action], object_to_idx: Dict[str, dict]
) -> List[NodeCandidate]:
    """Host nodes that appear as `source_host` in the candidate set."""
    valid = {
        action.parameters["source_host"]
        for action in candidates
        if "source_host" in action.parameters
    }
    return [
        (obj, idx) for obj, idx in object_to_idx.get("host", {}).items() if obj in valid
    ]


def target_candidates(
    candidates: Sequence[Action],
    action_type: ActionType,
    object_to_idx: Dict[str, dict],
) -> Tuple[List[NodeCandidate], Optional[str], Optional[str]]:
    """Nodes valid as the primary target, plus the parameter and node type."""
    parameter = ACTION_TARGET_PARAM.get(action_type)
    node_type = ACTION_TARGET_NODE_TYPE.get(action_type)
    if parameter is None or node_type is None:
        return [], parameter, node_type

    valid = {
        action.parameters[parameter]
        for action in candidates
        if parameter in action.parameters
    }
    return (
        [
            (obj, idx)
            for obj, idx in object_to_idx.get(node_type, {}).items()
            if obj in valid
        ],
        parameter,
        node_type,
    )


def secondary_candidates(
    candidates: Sequence[Action],
    action_type: ActionType,
    object_to_idx: Dict[str, dict],
) -> Tuple[List[NodeCandidate], Optional[str], Optional[str]]:
    """Nodes valid as the secondary parameter, for the types that use step 4."""
    if action_type not in SECONDARY_HEAD_ACTION_TYPES:
        return [], None, None

    parameter = ACTION_SECONDARY_PARAM.get(action_type)
    node_type = ACTION_SECONDARY_NODE_TYPE.get(action_type)
    if parameter is None or node_type is None:
        return [], parameter, node_type

    valid = {
        action.parameters[parameter]
        for action in candidates
        if parameter in action.parameters
    }
    return (
        [
            (obj, idx)
            for obj, idx in object_to_idx.get(node_type, {}).items()
            if obj in valid
        ],
        parameter,
        node_type,
    )


def narrow(
    candidates: Sequence[Action], parameter: str, value: Any
) -> List[Action]:
    """Keep candidates whose `parameter` equals `value`."""
    return [
        action for action in candidates if action.parameters.get(parameter) == value
    ]


def head_targets(
    action: Action,
    candidates: Sequence[Action],
    object_to_idx: Dict[str, dict],
) -> Optional[HeadTargets]:
    """Indices that reproduce `action`, replaying the decoder's narrowing.

    Returns `None` when the action is not reachable through the factorization
    from this candidate set — the caller should drop that sample rather than
    train on an unreachable target.
    """
    if action.type not in ACTION_TYPE_LIST:
        return None
    mask = type_mask(candidates)
    type_index = ACTION_TYPE_LIST.index(action.type)
    if not mask[type_index]:
        return None

    remaining = [candidate for candidate in candidates if candidate.type == action.type]

    # ── source host ──────────────────────────────────────────────────────────
    source_index: Optional[int] = None
    sources = source_candidates(remaining, object_to_idx)
    if sources:
        wanted = action.parameters.get("source_host")
        keys = [obj for obj, _ in sources]
        if wanted not in keys:
            return None
        source_index = keys.index(wanted)
        remaining = narrow(remaining, "source_host", wanted)

    # ── primary target ───────────────────────────────────────────────────────
    targets, target_param, _ = target_candidates(remaining, action.type, object_to_idx)
    if target_param is None or not targets:
        return None
    wanted = action.parameters.get(target_param)
    keys = [obj for obj, _ in targets]
    if wanted not in keys:
        return None
    target_index = keys.index(wanted)
    remaining = narrow(remaining, target_param, wanted)

    # ── secondary parameter ──────────────────────────────────────────────────
    secondary_index: Optional[int] = None
    secondaries, secondary_param, _ = secondary_candidates(
        remaining, action.type, object_to_idx
    )
    if secondary_param is not None and secondaries:
        wanted = action.parameters.get(secondary_param)
        keys = [obj for obj, _ in secondaries]
        if wanted not in keys:
            return None
        secondary_index = keys.index(wanted)
        remaining = narrow(remaining, secondary_param, wanted)

    if len(remaining) != 1:
        # The factorization does not identify a single action, so the label
        # cannot be attributed to one set of head choices.
        return None

    return HeadTargets(
        type_index=type_index,
        source_index=source_index,
        target_index=target_index,
        secondary_index=secondary_index,
    )


def decode_targets(
    targets: HeadTargets,
    candidates: Sequence[Action],
    object_to_idx: Dict[str, dict],
) -> Optional[Action]:
    """Rebuild the action a set of head indices selects.

    The inverse of `head_targets`, used to prove the two agree.
    """
    action_type = ACTION_TYPE_LIST[targets.type_index]
    remaining = [candidate for candidate in candidates if candidate.type == action_type]

    sources = source_candidates(remaining, object_to_idx)
    if targets.source_index is not None:
        if targets.source_index >= len(sources):
            return None
        remaining = narrow(remaining, "source_host", sources[targets.source_index][0])

    picks, target_param, _ = target_candidates(remaining, action_type, object_to_idx)
    if target_param is None or targets.target_index is None:
        return None
    if targets.target_index >= len(picks):
        return None
    remaining = narrow(remaining, target_param, picks[targets.target_index][0])

    secondaries, secondary_param, _ = secondary_candidates(
        remaining, action_type, object_to_idx
    )
    if targets.secondary_index is not None and secondary_param is not None:
        if targets.secondary_index >= len(secondaries):
            return None
        remaining = narrow(
            remaining, secondary_param, secondaries[targets.secondary_index][0]
        )

    return remaining[0] if len(remaining) == 1 else None
