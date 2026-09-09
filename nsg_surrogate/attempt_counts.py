"""Counts of attempted actions within one episode, per target.

Behaviour matches `sgrl_netsec/attempt_counts.py`, so the node features a
simulator-trained checkpoint expects are reproduced exactly. The counts describe
the agent's own interaction history rather than the environment, so they are held
by the agent and passed to the encoder rather than derived from the state.

They matter more in the emulated range than in the simulator. The state graph is
rebuilt on a timer rather than after each action, so an action is not guaranteed
to be followed by a state that reflects it; `record()` is then the only evidence
that the action was attempted.
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

    def snapshot(self) -> "AttemptCounts":
        """An independent copy, for recording history as of one step."""
        return AttemptCounts(
            scan=dict(self.scan),
            findservices=dict(self.findservices),
            finddata=dict(self.finddata),
            exploit=dict(self.exploit),
            exfil=dict(self.exfil),
        )

    def total(self) -> int:
        return sum(
            sum(counter.values())
            for counter in (self.scan, self.findservices, self.finddata, self.exploit, self.exfil)
        )
