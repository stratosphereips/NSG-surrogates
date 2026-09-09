"""Which parameter each of the four heads selects, per action type.

The tables follow `sgrl_netsec/policy_netsec.py`. Holding them as data rather
than as branches inside the policy is what allows four heads to produce five
differently shaped actions, and it means the action vocabulary is described in
one place.

    ScanNetwork      source -> target_network
    FindServices     source -> target_host
    FindData         source -> target_host
    ExploitService   source -> target_host   (service filled deterministically)
    ExfiltrateData   source -> data          -> target_host (destination)
"""

from typing import Dict, List

from netsecgame.game_components import ActionType

ACTION_TYPE_IDS: Dict[ActionType, int] = {
    ActionType.ScanNetwork: 0,
    ActionType.FindServices: 1,
    ActionType.FindData: 2,
    ActionType.ExploitService: 3,
    ActionType.ExfiltrateData: 4,
}

#: Index in this list is the integer the policy's type head emits.
ACTION_TYPE_LIST: List[ActionType] = sorted(
    ACTION_TYPE_IDS.keys(), key=lambda action_type: ACTION_TYPE_IDS[action_type]
)

#: Step 3 — the primary target parameter each action type carries.
ACTION_TARGET_PARAM: Dict[ActionType, str] = {
    ActionType.ScanNetwork: "target_network",
    ActionType.FindServices: "target_host",
    ActionType.ExploitService: "target_host",
    ActionType.FindData: "target_host",
    ActionType.ExfiltrateData: "data",
}

#: Step 3 — which node type holds that target in the graph.
ACTION_TARGET_NODE_TYPE: Dict[ActionType, str] = {
    ActionType.ScanNetwork: "network",
    ActionType.FindServices: "host",
    ActionType.ExploitService: "host",
    ActionType.FindData: "host",
    ActionType.ExfiltrateData: "data",
}

#: Step 4 — only ExfiltrateData uses this head in the blackbox agent; the
#: ExploitService entry is kept because the schema declares it and a future
#: policy may score services instead of canonicalizing them away.
ACTION_SECONDARY_PARAM: Dict[ActionType, str] = {
    ActionType.ExploitService: "target_service",
    ActionType.ExfiltrateData: "target_host",
}

ACTION_SECONDARY_NODE_TYPE: Dict[ActionType, str] = {
    ActionType.ExploitService: "service",
    ActionType.ExfiltrateData: "host",
}

#: Action types whose step-4 head the policy actually runs.
SECONDARY_HEAD_ACTION_TYPES = frozenset({ActionType.ExfiltrateData})
