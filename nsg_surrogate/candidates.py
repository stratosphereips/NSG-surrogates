"""Candidate action enumeration for a projected `GameState`.

The factored policy never scores a free-form action: every head is masked
against the candidate set, so this module defines what the surrogate is allowed
to emit. `netsecgame`'s own `generate_valid_actions` is the base, so the
surrogate's action space is the one an agent trained in the simulator learned
against — that identity is the premise of porting a policy out to the real
range.

One documented correction is applied on top: the upstream generator emits
`ExfiltrateData` from hosts the agent does not control, which the game will not
execute (see `enumerate_actions`). Knowing where data is does not mean being
able to take it.

Not representable at all, and therefore not in the candidate set: exfiltrating
data from a *controlled* host without having discovered it first — the blind
`scp /etc/shadow` that a real operator performs routinely. NSG enumerates
exfiltration over `known_data` only, so the action has no expression until the
vocabulary is extended. `labeling` records those attempts as unmappable rather
than inventing candidates for them.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from netsecgame.game_components import Action, ActionType, GameState
from netsecgame.utils.utils import generate_valid_actions

#: What `nsg-action-translator` can currently do with an emitted action.
#: Source: its README action-capability table. An action the surrogate emits
#: outside the `live` set is a well-formed NSG action that the real environment
#: will not carry out, so the PoC has to measure how often that happens.
TRANSLATOR_SUPPORT: Dict[ActionType, str] = {
    ActionType.ScanNetwork: "live",
    ActionType.FindServices: "live",
    ActionType.FindData: "unsupported",
    ActionType.ExploitService: "blocked",
    ActionType.ExfiltrateData: "blocked",
    ActionType.BlockIP: "unsupported",
}

EXECUTABLE_ACTION_TYPES = frozenset(
    action_type for action_type, status in TRANSLATOR_SUPPORT.items() if status == "live"
)


def action_key(action: Action) -> Tuple[ActionType, Tuple[Tuple[str, Any], ...]]:
    """Order-independent identity for an action.

    `Action.as_dict` (and therefore `to_json`) serialises parameters in
    insertion order, so two actions that are semantically the same compare
    unequal as JSON when their keys were assigned in a different order — which
    is exactly what happens between `generate_valid_actions` and any action
    built by hand. Comparing a sorted tuple of parameters instead uses the
    parameter objects' own equality (`IP`, `Network`, `Service` and `Data` are
    all frozen and hashable), so the key is both correct and usable in a set.
    """
    return (action.type, tuple(sorted(action.parameters.items(), key=lambda item: item[0])))


def action_sort_key(action: Action) -> str:
    """Stable ordering key. Set iteration order is random; graphs are not."""
    action_type, parameters = action_key(action)
    return f"{action_type.name}|" + ";".join(f"{name}={value}" for name, value in parameters)


def enumerate_actions(
    state: GameState,
    include_blocks: bool = False,
    canonicalize_exploit_services: bool = True,
    require_controlled_exfil_source: bool = True,
) -> List[Action]:
    """All NSG-valid actions for `state`, in deterministic order.

    `require_controlled_exfil_source` corrects the upstream generator. NSG's
    `generate_valid_actions` builds `ExfiltrateData` by iterating
    `state.known_data` for the source host and never checks that the agent
    controls it, so it emits exfiltrations *from* hosts the agent has only
    learned about — which the game itself will not carry out. Knowing that data
    sits on a host is not the same as being able to take it. The correction is
    applied by default and is a deliberate, documented divergence from
    upstream; pass `False` to reproduce the raw generator.
    """
    actions = sorted(generate_valid_actions(state, include_blocks=include_blocks), key=action_sort_key)
    if require_controlled_exfil_source:
        actions = drop_uncontrolled_exfil_sources(actions, state)
    if canonicalize_exploit_services:
        actions = canonicalize_exploits(actions)
    return actions


def drop_uncontrolled_exfil_sources(
    actions: Iterable[Action], state: GameState
) -> List[Action]:
    """Remove exfiltrations whose source host the agent does not control."""
    return [
        action
        for action in actions
        if action.type != ActionType.ExfiltrateData
        or action.parameters.get("source_host") in state.controlled_hosts
    ]


def canonicalize_exploits(actions: Iterable[Action]) -> List[Action]:
    """Collapse `ExploitService` to one canonical service per (source, target).

    Lifted from `blackbox_pure_gnn_agent._canonicalize_valid_actions`: the
    simulator's exploit outcome does not depend on which valid service is named,
    so exploiting is treated as a host-level decision and the service is filled
    with the lexicographically smallest valid one.

    This assumption is simulator-specific and does not hold in a real range,
    where the service decides whether the exploit exists at all. It is kept here
    so a policy trained in the simulator can be run unchanged, and flagged as a
    known divergence rather than silently inherited.
    """
    reduced: List[Action] = []
    exploit_index: Dict[Tuple[object, object], int] = {}

    for action in actions:
        if action.type != ActionType.ExploitService:
            reduced.append(action)
            continue

        key = (action.parameters.get("source_host"), action.parameters.get("target_host"))
        existing_idx = exploit_index.get(key)
        if existing_idx is None:
            exploit_index[key] = len(reduced)
            reduced.append(action)
            continue

        existing = reduced[existing_idx]
        if str(action.parameters.get("target_service")) < str(
            existing.parameters.get("target_service")
        ):
            reduced[existing_idx] = action

    return reduced


def breakdown(actions: Iterable[Action]) -> Dict[str, int]:
    """Count candidates per action type, for reports and masks."""
    counts: Dict[str, int] = {}
    for action in actions:
        key = action.type.name
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def executable(actions: Iterable[Action]) -> List[Action]:
    """Keep only actions the translator can currently execute for real."""
    return [action for action in actions if action.type in EXECUTABLE_ACTION_TYPES]


def support_status(action: Action) -> str:
    """`live`, `unsupported`, `blocked`, or `unknown` for one action."""
    return TRANSLATOR_SUPPORT.get(action.type, "unknown")


def describe(action: Action) -> str:
    """One-line human form: `ExploitService src=1.2.3.4 target_host=5.6.7.8 ...`."""
    parameters = " ".join(
        f"{key}={value}" for key, value in sorted(action.parameters.items(), key=lambda kv: kv[0])
    )
    return f"{action.type.name} {parameters}".strip()


def find_action(
    actions: Iterable[Action], action_type: ActionType, **parameters: object
) -> Optional[Action]:
    """First candidate matching a type and a subset of parameters, else None."""
    for action in actions:
        if action.type != action_type:
            continue
        if all(action.parameters.get(key) == value for key, value in parameters.items()):
            return action
    return None


def contains(actions: Iterable[Action], action: Action) -> bool:
    """Is this exact action in the candidate set, ignoring parameter order?"""
    wanted = action_key(action)
    return any(action_key(candidate) == wanted for candidate in actions)
