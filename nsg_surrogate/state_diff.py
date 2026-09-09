"""The difference between two projected states, in NetSecGame's own categories.

The state creator computes a delta over the recorded graph. That delta is useful
as evidence, but it is not the right basis for labelling: after scoping,
confidence thresholds and data filtering, a change in the recorded graph may not
appear in the projected `GameState`, and a change in the projection may not
correspond to a single recorded change. Whether observed behaviour is
expressible as a NetSecGame action depends on the change in NetSecGame's six
knowledge categories, which is what this module computes.

Each field corresponds to exactly one attacker action type, which is what makes
effect-based labelling possible:

    new_networks / new_hosts   ScanNetwork
    new_services               FindServices
    new_controlled             ExploitService
    new_data                   FindData
    relocated_data             ExfiltrateData
    new_blocks                 (defender vocabulary — no attacker action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

from netsecgame.game_components import Data, GameState, IP, Network


@dataclass(frozen=True)
class StateDiff:
    """What the agent learned or gained between two projected states."""

    new_networks: FrozenSet[Network] = frozenset()
    new_hosts: FrozenSet[IP] = frozenset()
    new_controlled: FrozenSet[IP] = frozenset()
    #: host -> services known on it now but not before
    new_services: Dict[IP, FrozenSet[object]] = field(default_factory=dict)
    #: host -> data known on it now but not before, and not known anywhere before
    new_data: Dict[IP, FrozenSet[Data]] = field(default_factory=dict)
    #: (data, origin host, destination host) for data that appeared on a host
    #: while already being known on another — the shape of an exfiltration.
    relocated_data: Tuple[Tuple[Data, IP, IP], ...] = ()
    #: host -> newly known blocked destinations
    new_blocks: Dict[IP, FrozenSet[IP]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not (
            self.new_networks
            or self.new_hosts
            or self.new_controlled
            or self.new_services
            or self.new_data
            or self.relocated_data
            or self.new_blocks
        )

    @property
    def changed_categories(self) -> Tuple[str, ...]:
        changed = []
        if self.new_networks:
            changed.append("known_networks")
        if self.new_hosts:
            changed.append("known_hosts")
        if self.new_controlled:
            changed.append("controlled_hosts")
        if self.new_services:
            changed.append("known_services")
        if self.new_data:
            changed.append("known_data")
        if self.relocated_data:
            changed.append("relocated_data")
        if self.new_blocks:
            changed.append("known_blocks")
        return tuple(changed)

    def summary(self) -> Dict[str, int]:
        return {
            "new_networks": len(self.new_networks),
            "new_hosts": len(self.new_hosts),
            "new_controlled": len(self.new_controlled),
            "new_services": sum(len(v) for v in self.new_services.values()),
            "new_data": sum(len(v) for v in self.new_data.values()),
            "relocated_data": len(self.relocated_data),
            "new_blocks": sum(len(v) for v in self.new_blocks.values()),
        }


def diff_states(before: GameState, after: GameState) -> StateDiff:
    """Compute the difference. Additions only: NetSecGame knowledge is monotonic.

    The recorded graph does lose facts — a file is deleted, a host stops
    responding — but no NetSecGame action reduces a state, so a decrease cannot
    be labelled and is not reported here. `lost_knowledge` measures those
    separately.
    """
    new_services: Dict[IP, FrozenSet[object]] = {}
    for host, services in after.known_services.items():
        gained = set(services) - set(before.known_services.get(host, set()))
        if gained:
            new_services[host] = frozenset(gained)

    data_hosts_before: Dict[Data, Set[IP]] = {}
    for host, items in before.known_data.items():
        for item in items:
            data_hosts_before.setdefault(item, set()).add(host)

    new_data: Dict[IP, FrozenSet[Data]] = {}
    relocated: List[Tuple[Data, IP, IP]] = []
    for host, items in after.known_data.items():
        gained = set(items) - set(before.known_data.get(host, set()))
        first_time, moved = set(), []
        for item in gained:
            origins = data_hosts_before.get(item, set())
            if origins:
                # Known elsewhere before, now also here: a copy was made.
                moved.extend((item, origin, host) for origin in sorted(origins, key=str))
            else:
                first_time.add(item)
        if first_time:
            new_data[host] = frozenset(first_time)
        relocated.extend(moved)

    new_blocks: Dict[IP, FrozenSet[IP]] = {}
    for host, blocked in after.known_blocks.items():
        gained = set(blocked) - set(before.known_blocks.get(host, set()))
        if gained:
            new_blocks[host] = frozenset(gained)

    return StateDiff(
        new_networks=frozenset(set(after.known_networks) - set(before.known_networks)),
        new_hosts=frozenset(set(after.known_hosts) - set(before.known_hosts)),
        new_controlled=frozenset(set(after.controlled_hosts) - set(before.controlled_hosts)),
        new_services=new_services,
        new_data=new_data,
        relocated_data=tuple(sorted(relocated, key=lambda triple: tuple(str(x) for x in triple))),
        new_blocks=new_blocks,
    )


def lost_knowledge(before: GameState, after: GameState) -> Dict[str, int]:
    """Categories that decreased. No NetSecGame action can produce a decrease."""
    lost_services = sum(
        len(set(services) - set(after.known_services.get(host, set())))
        for host, services in before.known_services.items()
    )
    lost_data = sum(
        len(set(items) - set(after.known_data.get(host, set())))
        for host, items in before.known_data.items()
    )
    return {
        "networks": len(set(before.known_networks) - set(after.known_networks)),
        "hosts": len(set(before.known_hosts) - set(after.known_hosts)),
        "controlled": len(set(before.controlled_hosts) - set(after.controlled_hosts)),
        "services": lost_services,
        "data": lost_data,
    }
