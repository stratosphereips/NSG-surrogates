"""Adapter that lets `nsg-action-translator` drive the surrogate.

The translator owns the other leg of the loop: it takes an NSG action and turns
it into a validated command plan it can execute in the container. It drives a
policy through the duck-typed interface declared in
`nsg_action_translator/policies/base.py`:

    reset(episode_id) -> None
    act(observation)  -> Action
    record_outcome(action, result) -> None
    info() -> dict

This class satisfies that interface without importing the translator, so this
repo has no hard dependency on it. Register it in the translator's policy
registry (which currently offers only `random`) to run the surrogate live.

One caveat worth keeping in view: the translator hands over an `Observation`
built by its own state provider, which carries a `GameState` and nothing else.
The provenance the encoder wants — real service ports, which hosts count as
external, docker node ids — does not survive that boundary, so pass it in at
construction time when the surrogate is driven this way.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from netsecgame.game_components import Action, Observation

from .candidates import enumerate_actions, support_status
from .policy import SurrogatePolicy
from .state_adapter import Provenance


class SurrogatePolicyAdapter:
    """`PolicyAdapter`-shaped wrapper around `SurrogatePolicy`."""

    def __init__(
        self,
        weights: Optional[str] = None,
        provenance: Optional[Provenance] = None,
        temperature: float = 0.1,
        legacy_service_port: bool = False,
    ):
        self.surrogate = SurrogatePolicy.load(weights, legacy_service_port=legacy_service_port)
        self.provenance = provenance
        self.temperature = temperature
        self.weights = weights
        self.episode_id: Optional[str] = None
        self.steps = 0
        self.emitted_unexecutable = 0

    def reset(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.steps = 0
        self.emitted_unexecutable = 0
        self.surrogate.reset()

    def act(self, observation: Observation) -> Action:
        actions = enumerate_actions(observation.state)
        if not actions:
            raise ValueError("no valid NSG action for the provided observation")

        decision = self.surrogate.act(
            observation.state,
            actions,
            provenance=self.provenance,
            temperature=self.temperature,
        )
        self.steps += 1
        if support_status(decision.action) != "live":
            # Tracked rather than suppressed: how often a simulator-shaped
            # policy asks for something the real range cannot do is one of the
            # numbers this proof of concept exists to produce.
            self.emitted_unexecutable += 1
        self.last_decision = decision
        return decision.action

    def record_outcome(self, action: Action, result: Dict[str, Any]) -> None:
        """Execution feedback. The attempt counter already advanced in `act`.

        The surrogate is stateless with respect to outcomes: NSG's own state
        update comes from the next state-creator rebuild, not from the command's
        stdout, and this runtime never interprets raw output into graph facts.
        """
        return None

    def info(self) -> Dict[str, Any]:
        return {
            "policy": "nsg_surrogate.FactoredGNNPolicy",
            "weights": self.weights or "<random-init>",
            "temperature": self.temperature,
            "episode_id": self.episode_id,
            "steps": self.steps,
            "emitted_unexecutable": self.emitted_unexecutable,
        }
