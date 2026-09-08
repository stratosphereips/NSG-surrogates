"""Surrogate agent: dockerized-environment state graph in, NetSecGame action out.

The package is deliberately split so that the mapping half runs without torch:

    state_adapter   NSG-state-creator graph.json  ->  netsecgame GameState
    candidates      GameState                     ->  valid parameterized actions
    encoder         GameState (+ counters)        ->  PyG HeteroData      (needs torch)
    policy          HeteroData + candidates       ->  one Action          (needs torch)

`nsg_surrogate.cli inspect` exercises the first two on a real observation
directory and reports what the mapping loses; that report is the point of the
proof of concept.
"""

__all__ = ["state_adapter", "candidates"]
