# NSG-surrogates

A surrogate agent for the dockerized NetSecGame loop: it reads the state graph
produced by **NSG-docker-state-creator** inside a real container and emits one
**parameterized NetSecGame action**, which **nsg-action-translator** turns into a
validated command plan and executes.

There are two distinct paths through this repo, and they map commands and
actions in **opposite directions**. Keeping them apart matters: only the first
one runs live.

**Runtime — state in, action out.** The surrogate decides; the translator turns
that decision into a command:

```
stratocyberlab container
        │
NSG-docker-state-creator ──► state/graph.json  (strategic level)
        │
        ▼
   ┌──────────────────── NSG-surrogates (this repo) ────────────────────┐
   │ state_adapter   graph.json      -> netsecgame GameState + report   │
   │ candidates      GameState       -> valid parameterized actions     │
   │ encoder         GameState       -> PyG HeteroData (4 node types)   │
   │ policy          graph + mask    -> one Action, four factored heads │
   └────────────────────────────────────────────────────────────────────┘
        │
        ▼  Action (NSG JSON)
nsg-action-translator ──► CommandPlan ──► container      [NSG action -> command]
```

**Offline — observed behaviour in, labelled pairs out.** No policy involved; this
is what the trajectories are for:

```
trajectory/sequence.jsonl + trajectory/states/*/graph.json
        │
        ▼
   ┌──────────────────── NSG-surrogates (this repo) ────────────────────┐
   │ dataset         walk transitions, dedupe collector records         │
   │ state_adapter   both state graphs  -> two GameStates               │
   │ state_diff      before vs after    -> NSG-level change             │
   │ labeling        change + argv      -> NSG Action + evidence        │
   └────────────────────────────────────────────────────────────────────┘
        │
        ▼
pairs.jsonl + unmappable.jsonl                    [command + effect -> NSG action]
```

The two mappings should agree where they overlap, and for the translator's two
implemented capabilities they do: it builds `nmap -sn` for `ScanNetwork` and
`nmap -sV` for `FindServices`, and `labeling.parse_command` recovers exactly
those action types from those flags (asserted in
`tests/test_labeling.py::TranslatorRoundTripTests`). The translator's plan
registry is the authoritative forward mapping, so `labeling.COMMAND_RULES`
should follow it as more capabilities are implemented there.

This is a proof of concept whose purpose is to find the problems in that
mapping. What it found is written up in
[`docs/poc-findings.md`](docs/poc-findings.md) — read that before trusting any
output. The longer-term goal (learning the docker environment into NetSecGame,
training agents in simulation, full knowledge-graph loop) is deliberately out of
scope here.

## Setup

```bash
conda activate nsg     # torch 2.14, torch-geometric 2.8, netsecgame 0.2.0 (editable
                       # from /opt/Agents/NetSecGame), netaddr, numpy
```

Nothing else is needed: `netsecgame` is already importable in that environment.
The state-mapping half (`state_adapter`, `candidates`) has no torch dependency
and runs under any Python 3.12+ with `netsecgame` on the path.

## Commands

```bash
# What does the mapping keep, and what does it throw away?
python -m nsg_surrogate inspect /opt/Agents/NSG-docker-state-creator/observation/manual-run \
    --scope 172.23.0.0/16 --external-host 10.9.9.9

# (state, action) pairs from a trajectory, with the label yield per source
python -m nsg_surrogate dataset <observation-dir> --scope 172.23.0.0/16 \
    --external-host 10.9.9.9 --out datasets/manual-run

# The projected NSG state, as the simulator would serialize it
python -m nsg_surrogate state <observation-dir> --scope 172.23.0.0/16

# One action, with the trace of what each head chose
python -m nsg_surrogate act <observation-dir> --scope 172.23.0.0/16 \
    --external-host 10.9.9.9 --weights best_blackbox_gnn.pth --temperature 0.1
```

`inspect --json` emits the same report as machine-readable JSON;
`inspect --show-drops N` lists individual dropped entities.

### Two flags that are not optional in practice

`--scope CIDR` restricts which hosts count as targets. The state creator records
every host the container ever contacted, so without a scope the projected NSG
state includes public internet hosts — 16 of 20 in the sample run — and NSG
treats every known host as attackable. See finding 1.

`--external-host IP` declares a host as outside the range. It replaces the
simulator's `is_private()` test (meaningless in an all-RFC1918 docker network)
and gives `ExfiltrateData` a destination, without which exfiltration is
unreachable. See findings 2 and 3.

## Module map

| Module | Needs torch | Responsibility |
|---|---|---|
| `state_adapter.py` | no | `graph.json` -> `GameState`, plus an `AdapterReport` of every dropped entity and forced choice, and a `Provenance` side channel (docker node ids, real service ports, external hosts) |
| `candidates.py` | no | valid-action enumeration via `netsecgame`'s own `generate_valid_actions`, plus one documented correction (no exfiltration from uncontrolled hosts), exploit canonicalization, translator support status |
| `state_diff.py` | no | NSG-level diff between two projected states; each category maps to exactly one action type |
| `labeling.py` | no | effect-primary + command-corroborating labels, the candidate gate, and the inventory of what NSG cannot express |
| `dataset.py` | no | trajectory walker, deduplication, and the self-contained dataset writer |
| `action_schema.py` | no | which parameter each factored head decides, per action type |
| `attempt_counts.py` | no | per-episode attempt counters, encoded into node features |
| `encoder.py` | yes | `GameState` -> `HeteroData`, feature-compatible with `sgrl_netsec` |
| `policy.py` | yes | `FactoredGNNPolicy` (parameter-identical to the simulator agent) and `SurrogatePolicy` |
| `translator_adapter.py` | yes | `PolicyAdapter`-shaped wrapper so `nsg-action-translator` can drive it |

## Tests

```bash
python -m unittest discover -s tests        # 81 tests
```

They are `unittest.TestCase` on purpose: the mapping tests run with the standard
library alone, so they work before torch is available, and pytest collects them
unchanged. `tests/test_policy.py` includes the load-bearing check that this
repo's policy is parameter-identical to `sgrl_netsec`'s `FactoredGNNPolicy`, so
a checkpoint trained in the simulator loads here without surgery.

## Dataset format

`dataset --out DIR` writes a directory whose source of truth is the state graph,
never a tensor and never a projected state:

```
pairs.jsonl        one row per labelled action
graphs/            the state graphs the rows reference, copied and hashed
unmappable.jsonl   observed behaviour NetSecGame cannot express
report.json        yields, label-source split, extension list
```

Each row carries `state_before` / `state_after` (id, relative graph path,
sha256), the `label` with its evidence, and a `projection_cache` block holding
the projected `GameState`, the adapter version and a hash of the adapter config.
That block is explicitly regenerable: a future adapter re-derives it from the
copied graph, so a mapping change re-projects the dataset instead of
invalidating it (`tests/test_dataset.py` asserts the round trip). Graphs are
copied in rather than referenced because observation directories are rotated and
rebuilt.

Labels come from two signals, effect first:

- **effect** — the NSG-level diff between the two projected states. Each of the
  six knowledge categories can only be grown by one action type, so the diff
  gives the type and most parameters, independently of which tool the operator
  used.
- **command** — `command.argv` on the attributed action records. It corroborates
  an effect label, recovers what a diff cannot see (which service was exploited,
  who acted), and labels actions that changed nothing — those matter, because a
  dataset of only successful transitions teaches the surrogate that every action
  pays off.

Measured on `manual-run` with `--scope 172.23.0.0/16`: 36,164 action records
collapse to 752 after deduplication, and yield **8 labels across 38
transitions** (4 effect, 3 command-only, 1 corroborated), against 315
off-vocabulary commands. See `docs/poc-findings.md` finding 16.

## Training

```bash
python -m nsg_surrogate train datasets/manual-run-strategic datasets/manual-run \
    --scope 172.23.0.0/16 --external-host 10.9.9.9 \
    --epochs 150 --lr 0.003 --out surrogate.pth --metrics metrics.json
```

Behaviour cloning over the four factored heads: masked cross-entropy against the
index `factorization.head_targets` says each head should have produced, with the
true prefix as conditioning (teacher forcing). Samples are weighted by label
confidence, so a `command`-only label at 0.5 counts half as much as a
corroborated one at 0.9.

Three properties the trainer enforces, each because getting it wrong would
produce a number that looks fine and means nothing:

- **One factorization, both directions.** `factorization.py` builds the per-head
  candidate lists used by *both* the decoder and the target builder. If the two
  enumerations drifted, training would optimise indices that decode to different
  actions at inference, and no loss curve would reveal it.
  `tests/test_training.py` round-trips every candidate in a state with all five
  action types through `head_targets` -> `decode_targets`.
- **Samples are re-projected from the copied graphs**, not read from
  `projection_cache`. The graph is the source of truth; the cache is only
  compared against, and a mismatch is reported as a stale dataset.
- **Splits are by run** (`--holdout-run`), never by row. Rows share states, so a
  row-level split leaks.

The reported metrics include two reference lines, because accuracy alone is
misleading at this scale: the **majority-class baseline** (a constant predictor)
and the **deterministic ceiling** — the best any state -> action function could
score, computed by grouping samples on what the encoder actually sees and
crediting the most frequent label per group. An accuracy at the ceiling means
every remaining error is a label conflict, not underfitting.

### What the 21 real pairs actually show

| | |
|---|---:|
| samples / distinct encoder inputs | 21 / 21 |
| majority-class baseline | 0.67 |
| deterministic ceiling | 1.00 |
| action accuracy (150 epochs, lr 3e-3) | 0.76 |
| action accuracy (300 epochs, lr 5e-3) | **1.00** (all four heads) |
| illegal actions emitted | 0 |

The first attempt scored exactly 0.7143 on every seed, which turned out to be
the *data* ceiling: without interaction history, 21 samples collapse to **8
distinct encoder inputs**, and 6 of them carry conflicting labels (the same
projected state labelled `FindData`, `FindServices` and `ScanNetwork` at
different points in the session). Replaying each run in trajectory order and
snapshotting `AttemptCounts` per step — which is what those counters were added
upstream for — makes all 21 inputs distinct and lifts the ceiling to 1.00.

With enough optimisation the model reaches the ceiling exactly — 21/21, every
head at 1.00, no illegal actions. That is the useful signal from this run: the
targets, the teacher forcing and the decoder all agree, because a factorization
mismatch anywhere would cap accuracy below 1.00 no matter how long it trained.

What it does *not* show is generalization. 21 samples over two runs of one
operator, evaluated in-sample, is a memorization test — a correctness check on
the pipeline, not a measurement of the surrogate.

## Status

Verified end to end on `NSG-docker-state-creator/observation/manual-run`: real
observation graph -> `GameState` -> hetero graph -> four factored heads -> NSG
action JSON the translator accepts.

Not yet done:

- No trained checkpoint exists on this machine, so transfer is proven
  structurally (weights load, shapes match) but not behaviourally.
- The only available observation runs are `operational` level; the intended
  input is `strategic`. A rebuild at that level is needed to confirm.
- The translator's policy registry offers only `random`, so
  `SurrogatePolicyAdapter` cannot yet be selected from its CLI.
