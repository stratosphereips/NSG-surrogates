"""The set of actions that are valid in a projected `GameState`.

Every head of the policy is masked against this set, so the module defines what
the surrogate may select. The set is produced by `netsecgame`'s own
`generate_valid_actions`, which keeps the action space identical to the one the
simulator's agents are trained against.

One filter is applied on top. The upstream generator currently offers
`ExfiltrateData` from hosts the agent does not control, and the game does not
carry those out; knowing where data resides does not imply being able to copy it
from there. The omission is reported upstream, and once it is fixed the filter
removes nothing (see `enumerate_actions`).

One action a real operator performs is absent from the set and cannot be added:
copying a file from a controlled host without having discovered it first, as in
`scp /etc/shadow`. NetSecGame enumerates exfiltration over `known_data`, so an
undiscovered file has no representation, and the policy's data head has no node
to score. `labeling` records such attempts as unmappable rather than
constructing candidates for them.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from netsecgame.game_components import Action, ActionType, GameState
from netsecgame.utils.utils import generate_valid_actions

def action_key(action: Action) -> Tuple[ActionType, Tuple[Tuple[str, Any], ...]]:
    """An identity for an action that does not depend on parameter order.

    `Action.as_dict`, and therefore `to_json`, serialises parameters in
    insertion order, so two equivalent actions compare unequal as JSON when
    their parameters were assigned in a different order. That occurs between
    `generate_valid_actions` and any action constructed here. Comparing a sorted
    tuple of parameters instead uses the parameter objects' own equality; `IP`,
    `Network`, `Service` and `Data` are frozen and hashable, so the key can also
    be used in a set.
    """
    return (action.type, tuple(sorted(action.parameters.items(), key=lambda item: item[0])))


def action_sort_key(action: Action) -> str:
    """A stable sort key. Set iteration order varies between processes."""
    action_type, parameters = action_key(action)
    return f"{action_type.name}|" + ";".join(f"{name}={value}" for name, value in parameters)


def enumerate_actions(
    state: GameState,
    include_blocks: bool = False,
    canonicalize_exploit_services: bool = True,
    require_controlled_exfil_source: bool = True,
) -> List[Action]:
    """All NSG-valid actions for `state`, in deterministic order.

    `require_controlled_exfil_source` excludes exfiltration actions whose source
    host the agent does not control. NetSecGame's `generate_valid_actions`
    builds `ExfiltrateData` by iterating `state.known_data` for the source host
    without checking control, so it offers actions the game will not carry out.
    That omission is reported upstream (`docs/poc-findings.md` finding 18); when
    it is fixed this filter removes nothing, so the default stays correct in
    both cases. Pass `False` to enumerate exactly what the installed
    `netsecgame` generates.
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

    The assumption is specific to the simulator and does not hold in a real
    range, where the service determines whether an exploit exists. It is
    retained so that a policy trained in the simulator runs unchanged, and is
    documented here rather than left implicit.
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
