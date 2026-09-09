"""Surrogate agents fitted to behaviour recorded in an emulated container.

The package is split so that the projection and labelling stages do not import
torch and can run wherever `netsecgame` is importable:

    state_adapter   state graph                -> netsecgame GameState
    candidates      GameState                  -> valid actions
    state_diff      two GameStates             -> change in NetSecGame terms
    labeling        change + recorded commands -> NetSecGame action
    dataset         a trajectory               -> labelled (state, action) rows

    encoder         GameState                  -> graph tensors   (needs torch)
    policy          graph tensors + candidates -> one action      (needs torch)
    training        labelled rows              -> fitted policy   (needs torch)
    nsg_agent       fitted policy              -> episodes against the game server

`python -m nsg_surrogate inspect` runs the projection alone and reports which
observed entities it kept, which it discarded, and why.
"""

__all__ = [
    "action_schema",
    "attempt_counts",
    "candidates",
    "dataset",
    "factorization",
    "labeling",
    "state_adapter",
    "state_diff",
]
