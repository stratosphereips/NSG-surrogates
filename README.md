# NSG-surrogates

Builds surrogate agents from recorded behaviour in an emulated network.

An operator (a person or a program) works inside an instrumented container.
NSG-docker-state-creator records what they did and how the observable world
changed. This repository converts those recordings into NetSecGame `(state,
action)` pairs, fits a policy to the pairs by supervised learning, and runs the
fitted policy as an ordinary NetSecGame agent. The result is a simulator agent
whose behaviour derives from measured activity in the emulated range rather than
from a hand-written baseline, so it can be evaluated on simulator scenarios and
used as an opponent when training other agents.

The repository stops at that boundary. Executing NetSecGame actions back in the
emulated range belongs to `nsg-action-translator`.

This is a proof of concept. Its purpose is to establish which parts of the
mapping work and which do not; the results are recorded in
[`docs/poc-findings.md`](docs/poc-findings.md), which should be read before any
output here is used.

## Terminology

| Term | Meaning in this repository |
|---|---|
| emulation | the instrumented container and the network it can reach |
| simulation | NetSecGame, driven by its coordinator process |
| state graph | `state/graph.json` from NSG-docker-state-creator: hosts, networks, services, data, blocks, with confidence values and evidence |
| projection | converting a state graph into a NetSecGame `GameState` |
| candidate set | the NetSecGame actions that are valid in a given state |
| label | the NetSecGame action attributed to one observed state transition |
| transition | a `(state_before, state_after)` pair named by the recorded action |

## Data flow

Recorded behaviour to fitted policy:

```
instrumented container
        │
NSG-docker-state-creator ──► trajectory/sequence.jsonl + states/*/graph.json
        │
        ▼
   ┌──────────────────── NSG-surrogates (this repo) ────────────────────┐
   │ dataset         walk transitions, deduplicate collector records    │
   │ state_adapter   state graph        -> GameState                    │
   │ state_diff      before vs after    -> change in GameState terms    │
   │ labeling        change + command   -> NetSecGame action            │
   │                                                                    │
   │ encoder         GameState          -> heterogeneous graph tensors   │
   │ training        pairs              -> four-head policy             │
   └────────────────────────────────────────────────────────────────────┘
        │                                            │
        ▼                                            ▼
   pairs.jsonl + unmappable.jsonl              models/surrogate.pth
```

Fitted policy to simulator agent:

```
models/surrogate.pth ──► nsg_agent (BaseAgent) ◄──socket──► NetSecGame server
```

The policy architecture and this second path follow
`sgrl_netsec/blackbox_pure_gnn_agent.py`, so results are reported in the same
form as the simulator's own agents. That is provenance, not a constraint: the
two are free to diverge, and parameter compatibility between them is not
maintained.

### The mapping this repository implements

`labeling` maps an observed command together with its observed effect to a
NetSecGame action. `nsg-action-translator` implements the opposite mapping, from
a NetSecGame action to an executable command, and is not a dependency here.

The two would need to agree only if a NetSecGame agent drove the emulated range
through the translator and this repository then labelled the resulting
recording. That is not the case today, and per finding 9 it cannot be for three
of the five action types. Consistency between them is therefore not asserted
here: doing so would require reading the translator's command plans, and copies
of them cannot detect a change on that side.

## Requirements

```bash
conda activate nsg
```

That environment provides everything: `netsecgame`, torch 2.14,
torch-geometric 2.8, netaddr and numpy.

`netsecgame` is an ordinary package dependency, published on PyPI:

```bash
pip install netsecgame                              # released version
pip install -e /path/to/NetSecGame                  # to track development
```

It is currently installed as an editable checkout (version 0.2.0), because the
package is under active development and this repository needs the current
version. Nothing here records where that checkout is; the package is imported by
name.

The projection and labelling modules (`state_adapter`, `candidates`,
`state_diff`, `labeling`, `dataset`) do not import torch and run under any
Python 3.12 or later with `netsecgame` importable.

### External dependencies

| Dependency | Kind | Extent |
|---|---|---|
| `netsecgame` | package import | `GameState`, `IP`, `Network`, `Service`, `Data`, `Action`, `ActionType`, `generate_valid_actions`, `BaseAgent`, `AgentRole` — nothing else |
| NSG-docker-state-creator | data format | the state graph and trajectory layout its output uses; no code is imported and no path is recorded |
| torch, torch-geometric, netaddr | package imports | the encoder, the policy and address arithmetic |

Knowledge of the state-creator format is confined to three modules:
`state_adapter` reads the state graph, `dataset` reads the trajectory layout,
and `labeling` reads the command fields of an action record. The format is
identified by `schema_version` (`nsg-state-graph/1.1`), which the projection
checks and reports, and `tests/test_dataset.py` constructs a synthetic
trajectory in that layout, so the assumptions are exercised without the other
repository being present.

## Running the pipeline

All commands run from the repository root. `OBS` is one observation directory.
Copy or freeze it first (`cp -a`, or stop `trajectory-monitor`): a directory
still being written by a running observer produces different counts on each
build.

```bash
export OBS=/opt/Agents/NSG-docker-state-creator/observation/manual-run-strategic
export SCOPE=172.23.0.0/16    # networks that are targets (finding 1)
export DROP=10.9.9.9          # exfiltration destination (finding 2)
```

**1. Report what the projection keeps and what it discards.** Writes nothing.

```bash
python -m nsg_surrogate inspect $OBS --scope $SCOPE --external-host $DROP
python -m nsg_surrogate inspect $OBS --scope $SCOPE --external-host $DROP --show-drops 20
python -m nsg_surrogate inspect $OBS --scope $SCOPE --external-host $DROP --json
```

**2. Print the projected state** in NetSecGame's own serialisation:

```bash
python -m nsg_surrogate state $OBS --scope $SCOPE --external-host $DROP
```

**3. Build a dataset.** Writes `datasets/<run_id>/`; `--no-write` reports
without writing. `--max-records-per-transition 0` removes the limit on how many
collector records are read per transition: exact, but slower on large runs.

```bash
python -m nsg_surrogate dataset $OBS --scope $SCOPE --external-host $DROP --max-records-per-transition 0
```

**4. Fit a policy.** Accepts one or more dataset directories and writes
`models/surrogate.pth` with `models/surrogate.metrics.json`.

```bash
python -m nsg_surrogate train datasets/* --scope $SCOPE --external-host $DROP --epochs 300 --lr 0.005
python -m nsg_surrogate train datasets/* --scope $SCOPE --external-host $DROP \
    --holdout-run manual-run --name surrogate-holdout
```

**5. Select one action** for a recorded state, with the choice made by each of
the four heads. Without `--weights` the policy is randomly initialised, which
still exercises the whole path.

```bash
python -m nsg_surrogate act $OBS --scope $SCOPE --external-host $DROP \
    --weights models/surrogate.pth --temperature 0.1
```

### Two options that should always be set

`--scope CIDR` limits which hosts are treated as targets. The state creator
records every host the container contacted, and NetSecGame treats every known
host as attackable. Without a scope, 16 of the 20 hosts projected from the
sample run are public internet addresses (finding 1).

`--external-host IP` declares a host to be outside the range. It replaces the
simulator's `is_private()` test, which cannot distinguish anything in an
all-RFC1918 container network, and it supplies the destination that
`ExfiltrateData` requires (findings 2 and 3).

### Options must match between steps

`--scope`, `--external-host`, `--min-confidence`, `--any-service-status`,
`--unconfirmed-data`, `--no-data-denylist`, `--include-inferred-networks` and
`--max-data-per-host` all change the projection. `train` re-projects from the
stored state graphs, so it must be given the same values as `dataset`. It
reports a count of rows whose re-projection differs from the one recorded in the
dataset.

## Running the agent in NetSecGame

**1. Start a game server.** This repository does not start one. From the
NetSecGame checkout:

```bash
cd /opt/Agents/NetSecGame
docker run -d --rm --name nsg-server \
  -v $(pwd)/examples/example_task_configuration.yaml:/netsecgame/netsecenv_conf.yaml \
  -v $(pwd)/logs:/netsecgame/logs \
  -p 9000:9000 stratosphereips/netsecgame

# or, without docker
python3 -m netsecgame.game.worlds.NetSecGame \
  --task_config=./examples/example_task_configuration.yaml --game_port=9000
```

The port the container publishes must match `--port` below.

**2. Play episodes.** Defaults are `models/surrogate.pth` and 10 episodes.

```bash
python -m nsg_surrogate play --host 127.0.0.1 --port 9000 --episodes 20
python -m nsg_surrogate play --port 9000 --episodes 20 --verbose
python -m nsg_surrogate play --port 9000 --episodes 100 --metrics models/play.json
python -m nsg_surrogate play --port 9000 --weights ""
```

Reported statistics follow the simulator agent's format:

```
win rate 0.0% ± 0.0 SE (0/20) | steps all 50.0 ± 0.0 | steps wins n/a | reward -60.00 ± 0.00
action types  {'FindData': 710, 'FindServices': 220, 'ScanNetwork': 70}
end reasons   {'AgentStatus.TimeoutReached': 20}
```

Each mean is reported with its dispersion: the standard error for the win rate,
the sample standard deviation for steps and reward. `steps wins` is the mean
over winning episodes only, and reads `n/a` when there were none rather than
`0.0`. `fallbacks` counts steps in which the factored representation could not
express the candidate set and the agent selected uniformly at random; it should
be zero.

### Measured behaviour of the current checkpoint

Twenty episodes on `example_task_configuration.yaml`, comparing the fitted
policy with a randomly initialised one:

| Action type | fitted policy | random initialisation | training pairs |
|---|---:|---:|---:|
| `FindData` | 710 (71%) | 268 (27%) | 14 (67%) |
| `FindServices` | 220 (22%) | 599 (60%) | 3 (14%) |
| `ScanNetwork` | 70 (7%) | 55 (6%) | 4 (19%) |
| `ExploitService` | 0 | 78 (8%) | 0 |
| `ExfiltrateData` | 0 | 0 | 0 |
| win rate | 0/20 | 0/20 | — |

The fitted policy's action distribution follows the training distribution rather
than the random baseline, so the supervised fit did transfer: the recorded
operator read local files and enumerated services, and the fitted policy does
the same. It selects `ExploitService` zero times, where the randomly initialised
policy selects it 78 times; having seen no example of that action type, the
fitted policy assigns it low probability.

Neither policy can win this scenario. The goal is defined in terms of
exfiltrated data, reaching it requires exploiting a host first, and the training
set contains no example of either action. A zero win rate is therefore the
expected result of fitting 21 pairs of discovery activity, and is not evidence
about the method.

### Options that change behaviour

| Option | Effect |
|---|---|
| `--temperature 0.1` (default) | Samples each head at low temperature. Taking the argmax of every head independently does not maximise the probability of the joint action, so the simulator agent samples as well. |
| `--greedy` | Takes each head's argmax. Deterministic; useful for reproducing a single episode. |
| `--raw-action-space` | Enumerates exactly what the installed `netsecgame` generates, including exfiltration actions whose source host is not controlled. Those are excluded by default because the game rejects them; the omission is reported upstream and, once fixed, this option changes nothing (finding 18). |
| `--external-host IP` | Declares a host outside the range. Without it the encoder falls back to `is_private()`, which is correct in the simulator and uninformative in a container network. |
| `--seed`, `--episodes`, `--metrics` | Random seed, episode count, and a path for the statistics as JSON. |

### Two limits of this path

- **Attacker role only.** The policy has no head for `BlockIP`, so it cannot act
  as `AgentRole.Defender`. Training a defender against the surrogate is
  supported; running the surrogate as a defender is not.
- **No learning.** `play` evaluates a fixed checkpoint.
  `SurrogateController.state_value()` exposes the value estimate from the shared
  network body for later use in a reinforcement learning loop.

## Output locations

| Path | Contents | Tracked in git |
|---|---|---|
| `datasets/<run_id>/` | `pairs.jsonl`, `graphs/`, `unmappable.jsonl`, `report.json` | no |
| `models/<name>.pth` | policy parameters (about 680 KB) | no |
| `models/<name>.metrics.json` | dataset summary, per-epoch losses, evaluation metrics | no |

Both directories are excluded from git and can be regenerated from the
observation runs. A dataset copies every state graph it references, and those
graphs are about 2 MB each (27 MB for the two runs used here). The locations can
be changed with `NSG_SURROGATE_DATASETS` and `NSG_SURROGATE_MODELS`, or per
command with `--out`.

## Modules

| Module | Needs torch | Responsibility |
|---|---|---|
| `state_adapter.py` | no | state graph to `GameState`, with a report of every discarded entity and forced choice, and a `Provenance` record of information `GameState` cannot hold (node identifiers, service ports, external hosts) |
| `candidates.py` | no | enumerates valid actions using `netsecgame`'s `generate_valid_actions`, with one documented correction and the exploit-service canonicalisation |
| `state_diff.py` | no | difference between two projected states, expressed in NetSecGame's knowledge categories |
| `labeling.py` | no | assigns a NetSecGame action to a transition from its effect and the recorded commands; records transitions it cannot express |
| `dataset.py` | no | walks a trajectory, deduplicates collector records, writes the dataset |
| `action_schema.py` | no | which parameter each of the four heads selects, per action type |
| `attempt_counts.py` | no | per-episode counts of attempted actions, encoded as node features |
| `factorization.py` | no | per-head candidate lists, shared by action selection and target construction |
| `encoder.py` | yes | `GameState` to heterogeneous graph tensors |
| `policy.py` | yes | `FactoredGNNPolicy` and the `SurrogatePolicy` wrapper |
| `training.py` | yes | supervised fitting, evaluation metrics, checkpointing |
| `nsg_agent.py` | yes | `SurrogateController` (action selection, no network connection) and `SurrogateAgent` (plays episodes against the game server) |

## Tests

```bash
python -m unittest discover -s tests        # 114 tests
```

They are written as `unittest.TestCase`: the projection and labelling tests need
only the standard library, so they run without torch, and pytest collects them
unchanged. No test depends on another repository being checked out.

## Dataset format

A dataset directory contains:

```
pairs.jsonl        one row per labelled action
graphs/            the state graphs the rows reference, copied and hashed
unmappable.jsonl   observed behaviour with no NetSecGame equivalent
report.json        counts, label sources, unmappable categories
```

Each row records `state_before` and `state_after` (identifier, relative path to
the copied graph, SHA-256), the label with its supporting evidence, and a
`projection_cache` block containing the projected `GameState`, the adapter
version and a hash of the adapter options.

The state graph is the authoritative content; `projection_cache` is derived and
can be recomputed. A later change to the projection therefore re-derives the
dataset instead of invalidating it, which `tests/test_dataset.py` verifies.
Graphs are copied into the dataset rather than referenced by path because
observation directories are rebuilt and rotated.

Labels come from two sources of evidence:

- **effect** — the difference between the two projected states. Each of
  NetSecGame's six knowledge categories can only be increased by one action
  type, so the difference determines the action type and most of its parameters,
  independently of which tool the operator used.
- **command** — the `argv` recorded for the actions attributed to the
  transition. It confirms an effect-derived label, supplies the two parameters
  the effect cannot determine (which service was exploited, and which host
  acted), and labels actions that produced no observable change. The last case
  matters: a dataset containing only state-changing transitions would represent
  every action as successful.

Both runs used here, built with `--scope 172.23.0.0/16 --external-host 10.9.9.9`:

| | `manual-run-strategic` | `manual-run` |
|---|---:|---:|
| detail level | strategic | operational |
| states / transitions | 12 / 22 | 39 / 65 |
| action records | 807 | 43,202 |
| records after deduplication | 271 | 752 |
| labels | 8 | 13 |
| transitions with no label | 14 | 52 |
| commands with no NetSecGame equivalent | 98 | 537 |

`manual-run` is still being written by a running observer, so its counts
increase between builds. `manual-run-strategic` is frozen and reproducible.

## Fitting the policy

```bash
python -m nsg_surrogate train datasets/manual-run-strategic datasets/manual-run \
    --scope 172.23.0.0/16 --external-host 10.9.9.9 --epochs 300 --lr 0.005
```

The method is supervised imitation of the labelled actions. For each sample, the
four heads are scored and each receives a cross-entropy loss against the index
that `factorization.head_targets` determines it should have produced. Each head
is conditioned on the correct earlier choices rather than its own (teacher
forcing), so a head is not trained to compensate for an earlier head's error.
Samples are weighted by label confidence: a command-only label at 0.5 
contributes half as much as a confirmed label at 0.9.

Three properties the trainer enforces:

- **One representation of the factorisation.** `factorization.py` builds the
  per-head candidate lists used both when selecting an action and when
  constructing training targets. If the two enumerations differed, training
  would optimise indices that correspond to different actions at selection
  time, and the loss would not reveal it. `tests/test_training.py` converts
  every candidate in a state containing all five action types to targets and
  back again.
- **Samples are re-projected from the copied graphs**, not read from
  `projection_cache`, so that a change to the projection changes the training
  data. The cached projection is used only for comparison.
- **Data is split by run** (`--holdout-run`), not by row. Several rows can share
  a state, so splitting by row would place the same state in both parts.

Reported metrics include two reference values, because accuracy alone is not
interpretable at this sample size:

- the **majority-class rate**, the accuracy of always predicting the most
  frequent action type;
- the **identifiability limit**, the highest accuracy any function of the
  encoder's input could reach, computed by grouping samples with identical
  encoder inputs and counting only the most frequent label in each group.
  Accuracy equal to this limit means the remaining errors are contradictory
  labels rather than underfitting.

### Result on the 21 available pairs

| | |
|---|---:|
| samples / distinct encoder inputs | 21 / 21 |
| majority-class rate | 0.67 |
| identifiability limit | 1.00 |
| action accuracy, 150 epochs, lr 3e-3 | 0.76 |
| action accuracy, 300 epochs, lr 5e-3 | 1.00 (all four heads) |
| invalid actions selected | 0 |

The first configuration reached 0.7143 for every random seed. That value was the
identifiability limit of the data, not a property of the model: without
interaction history the 21 samples reduce to 8 distinct encoder inputs, and 6 of
those carry more than one distinct label, because the same projected state was
labelled `FindData`, `FindServices` and `ScanNetwork` at different points in the
session. Replaying each run in recorded order and attaching the accumulated
`AttemptCounts` to each sample makes all 21 inputs distinct and raises the limit
to 1.00.

Given enough optimisation the model then reaches that limit exactly. This
confirms that target construction, teacher forcing and action selection are
consistent: any disagreement between them would bound accuracy below 1.00
regardless of training length. It does not measure generalisation. Twenty-one
samples from two sessions of one operator, evaluated on the training data, is a
consistency check on the implementation.

## Current limitations

- 21 labelled pairs covering three of the five action types. `ExploitService`
  and `ExfiltrateData` have no examples, so the checkpoint demonstrates the
  implementation rather than useful behaviour.
- One frozen observation run at the intended `strategic` detail level. The other
  measurements come from an `operational` graph.
- Label confidence is bounded by attribution quality in the recordings: no state
  change in either run could be attributed to a single command
  (`causal_isolation` is false throughout), and 19 of 807 records in the
  strategic run come from the high-confidence interactive-shell collector
  (findings 16 and 19).
- The host that issued an action is not recorded. The observed container is
  preferred, and the choice is noted whenever more than one host is controlled;
  actions taken through a remote-execution wrapper are reported rather than
  attributed (finding 21).
