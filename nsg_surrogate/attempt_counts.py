"""Per-episode, per-action attempt counters.

Lifted unchanged in behaviour from `sgrl_netsec/attempt_counts.py` so that the
node-feature layout a simulator-trained checkpoint expects is reproduced
exactly. The counts belong to the agent's interaction history, not to the
environment, so the surrogate owns them rather than the state adapter.

In the real range there is one wrinkle: the state creator's graph is rebuilt on
a timer, not per action, so a surrogate step is not guaranteed to be followed by
a fresh state. `record()` is therefore the only reliable record that an action
was attempted at all.
"""

from dataclasses import dataclass, field
from typing import Dict

from netsecgame.game_components import Action, ActionType, Data, IP, Network


@dataclass
class AttemptCounts:
    scan: Dict[Network, int] = field(default_factory=dict)
    findservices: Dict[IP, int] = field(default_factory=dict)
    finddata: Dict[IP, int] = field(default_factory=dict)
    exploit: Dict[IP, int] = field(default_factory=dict)
    exfil: Dict[Data, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.scan.clear()
        self.findservices.clear()
        self.finddata.clear()
        self.exploit.clear()
        self.exfil.clear()

    def record(self, action: Action) -> None:
        parameters = action.parameters
        counter, key = {
            ActionType.ScanNetwork: (self.scan, parameters.get("target_network")),
            ActionType.FindServices: (self.findservices, parameters.get("target_host")),
            ActionType.FindData: (self.finddata, parameters.get("target_host")),
            ActionType.ExploitService: (self.exploit, parameters.get("target_host")),
            ActionType.ExfiltrateData: (self.exfil, parameters.get("data")),
        }.get(action.type, (None, None))

        if counter is not None and key is not None:
            counter[key] = counter.get(key, 0) + 1

    def total(self) -> int:
        return sum(
            sum(counter.values())
            for counter in (self.scan, self.findservices, self.finddata, self.exploit, self.exfil)
        )
