# NSG-surrogates

Builds surrogate agents from **emulation trajectories**. It turns behaviour
observed inside a real container into NetSecGame-space `(state, action)` pairs,
clones a policy from them, and runs that policy as an ordinary NetSecGame agent —
so behaviour recorded in the emulated range can be replayed, measured, and
trained against in simulation.

Scope stops there. Executing NSG actions back in the emulated range is
`nsg-action-translator`'s job, not this repo's.

This is a proof of concept whose purpose is to find the problems in that
mapping. What it found is in [`docs/poc-findings.md`](docs/poc-findings.md) —
read that before trusting any output.

Emulation traces in, cloned policy out:

```
stratocyberlab container
        │
NSG-docker-state-creator ──► trajectory/sequence.jsonl + states/*/graph.json
        │
        ▼
   ┌──────────────────── NSG-surrogates (this repo) ────────────────────┐
   │ dataset         walk transitions, dedupe collector records         │
   │ state_adapter   state graph        -> netsecgame GameState         │
   │ state_diff      before vs after    -> NSG-level change             │
   │ labeling        change + argv      -> NSG Action + evidence        │
   │                                                                    │
   │ encoder         GameState          -> PyG HeteroData               │
   │ training        pairs              -> cloned four-head policy      │
   └────────────────────────────────────────────────────────────────────┘
        │                                            │
        ▼                                            ▼
   pairs.jsonl + unmappable.jsonl              models/surrogate.pth
```

and that policy plays in the simulator over the coordinator's socket, the way
`sgrl_netsec/blackbox_pure_gnn_agent.py` does:

```
models/surrogate.pth ──► nsg_agent (BaseAgent) ◄──socket──► NetSecGame server
```

Verified: 20 episodes against a live server, with a random-init control (see
[Play the surrogate in NetSecGame](#play-the-surrogate-in-netsecgame)).

### The mapping this repo owns

`labeling` maps **command + observed effect -> NSG action**. That is the inverse
of what `nsg-action-translator` does (NSG action -> command), and the two agree
where they overlap: the translator builds `nmap -sn` for `ScanNetwork` and
`nmap -sV` for `FindServices`, and `labeling.parse_command` recovers exactly
those action types from those flags — asserted in
`tests/test_labeling.py::TranslatorRoundTripTests`. That test is the guard
against the two drifting apart; nothing else here depends on the translator.

## Setup

```bash
conda activate nsg     # torch 2.14, torch-geometric 2.8, netsecgame 0.2.0 (editable
                       # from /opt/Agents/NetSecGame), netaddr, numpy
```

Nothing else is needed: `netsecgame` is already importable in that environment.
The state-mapping half (`state_adapter`, `candidates`) has no torch dependency
and runs under any Python 3.12+ with `netsecgame` on the path.

## Run the whole pipeline

Every command below is copy-pasteable from the repo root with `conda activate
nsg`. `OBS` is one frozen observation run; **freeze it first** (`cp -a`, or stop
`trajectory-monitor`) because a live run is still being appended to and its
label counts drift between builds.

```bash
export OBS=/opt/Agents/NSG-docker-state-creator/observation/manual-run-strategic
export SCOPE=172.23.0.0/16    # in-range targets only (finding 1)
export DROP=10.9.9.9          # exfiltration destination (finding 2)
```

**1. See what the mapping keeps and what it discards.** Nothing is written.

```bash
python -m nsg_surrogate inspect $OBS --scope $SCOPE --external-host $DROP
python -m nsg_surrogate inspect $OBS --scope $SCOPE --external-host $DROP --show-drops 20   # individual entities
python -m nsg_surrogate inspect $OBS --scope $SCOPE --external-host $DROP --json            # machine-readable
```

**2. Look at the projected NSG state itself**, serialised exactly as the
simulator would:

```bash
python -m nsg_surrogate state $OBS --scope $SCOPE --external-host $DROP
```

**3. Build the dataset.** Writes to `datasets/<run_id>/` by default; add
`--no-write` for the report alone. `--max-records-per-transition 0` disables the
per-transition cap on how many collector records are loaded (exact but slower —
`manual-run` has ~39k records).

```bash
python -m nsg_surrogate dataset $OBS --scope $SCOPE --external-host $DROP --max-records-per-transition 0
```

**4. Train.** Takes one or more dataset directories; writes
`models/surrogate.pth` and `models/surrogate.metrics.json` by default.

```bash
python -m nsg_surrogate train datasets/* --scope $SCOPE --external-host $DROP --epochs 300 --lr 0.005
python -m nsg_surrogate train datasets/* --scope $SCOPE --external-host $DROP \
    --holdout-run manual-run --name surrogate-holdout    # honest split, when you have runs to spare
```

**5. Emit one action** for the current state, with the trace of what each head
chose. Omit `--weights` for a randomly initialised policy, which is still useful
for exercising the whole path:

```bash
python -m nsg_surrogate act $OBS --scope $SCOPE --external-host $DROP \
    --weights models/surrogate.pth --temperature 0.1
```

### Two flags that are not optional in practice

`--scope CIDR` restricts which hosts count as targets. The state creator records
every host the container ever contacted, so without a scope the projected NSG
state includes public internet hosts — 16 of 20 in the sample run — and NSG
treats every known host as attackable. See finding 1.

`--external-host IP` declares a host as outside the range. It replaces the
simulator's `is_private()` test (meaningless in an all-RFC1918 docker network)
and gives `ExfiltrateData` a destination, without which exfiltration is
unreachable. See findings 2 and 3.

Any flag in steps 1–4 that changes the projection (`--scope`, `--external-host`,
`--min-confidence`, `--any-service-status`, `--unconfirmed-data`,
`--no-data-denylist`, `--include-inferred-networks`, `--max-data-per-host`) must
match between `dataset` and `train`: the trainer re-projects from the stored
graphs, and it warns when the result differs from what the dataset recorded.

## Play the surrogate in NetSecGame

The return leg of the emulation -> simulation -> emulation loop: the same policy
trained on real container states runs as an ordinary NSG agent over the
coordinator's socket, the way `sgrl_netsec/blackbox_pure_gnn_agent.py` does. That
makes the surrogate both **measurable** on simulator scenarios and **usable as an
opponent** for training other agents in NetSecGame.

**1. Start a game server** (not started by this repo). From the NetSecGame
checkout:

```bash
cd /opt/Agents/NetSecGame
docker run -d --rm --name nsg-server \
  -v $(pwd)/examples/example_task_configuration.yaml:/netsecgame/netsecenv_conf.yaml \
  -v $(pwd)/logs:/netsecgame/logs \
  -p 9000:9000 stratosphereips/netsecgame

# or without docker
python3 -m netsecgame.game.worlds.NetSecGame \
  --task_config=./examples/example_task_configuration.yaml --game_port=9000
```

**2. Play.** Defaults to `models/surrogate.pth` and 10 episodes:

```bash
cd /opt/Agents/NSG-surrogates
python -m nsg_surrogate play --host 127.0.0.1 --port 9000 --episodes 20
python -m nsg_surrogate play --episodes 20 --verbose            # every action
python -m nsg_surrogate play --episodes 100 --metrics models/play.json
python -m nsg_surrogate play --weights ""                       # random policy, exercises the path
```

Output is the same shape the simulator agent reports, so numbers are directly
comparable. This is the real first run, 20 episodes on
`example_task_configuration.yaml`:

```
win rate 0.0% ± 0.0 SE (0/20) | steps all 50.0 ± 0.0 | steps wins n/a | reward -60.00 ± 0.00
action types  {'FindData': 710, 'FindServices': 220, 'ScanNetwork': 70}
end reasons   {'AgentStatus.TimeoutReached': 20}
```

Spread travels with every mean on purpose: a win rate from 20 episodes is not a
measurement without its standard error, and `steps wins` reads `n/a` rather than
`0.0` when nothing was won. `fallbacks` counts steps where the factorization
could not represent the candidate set and the agent had to act uniformly — it
should be 0.

### First live run: the clone reproduces its operator, including the gaps

Run against a live server on port 9099, 20 episodes each, trained checkpoint
versus a randomly initialised policy as the control:

| Action type | trained surrogate | random init | training pairs |
|---|---:|---:|---:|
| `FindData` | 710 (71%) | 268 (27%) | 14 (67%) |
| `FindServices` | 220 (22%) | 599 (60%) | 3 (14%) |
| `ScanNetwork` | 70 (7%) | 55 (6%) | 4 (19%) |
| `ExploitService` | **0** | 78 (8%) | 0 |
| `ExfiltrateData` | 0 | 0 | 0 |
| win rate | 0/20 | 0/20 | — |

The action mix tracks the *training distribution*, not the random baseline, so
the behaviour cloning transferred something real: this operator read local files
and enumerated services, and so does the clone. It is also the sharpest possible
demonstration of the gap — the surrogate emits `ExploitService` **zero** times
where the untrained policy tries it 78 times. Having never seen an exploit, the
clone has learned to avoid the one action type that would let it progress.

Neither policy wins, and neither can: the scenario's goal is exfiltrated data,
reaching it requires exploiting a host first, and no training pair contains
either action. A 0% win rate here is the expected, correct outcome of cloning 21
discovery-only steps — not a bug in the agent, and not a ceiling on the method.

### Flags that change behaviour

| Flag | Effect |
|---|---|
| `--temperature 0.1` (default) | Low-temperature sampling. Pure argmax composes badly in a factored action space — each head's independent best does not make the best action — which is why the simulator agent samples too. |
| `--greedy` | Take each head's argmax anyway. Deterministic, useful for reproducing one episode. |
| `--raw-action-space` | Keep NSG's exfiltrations from *uncontrolled* source hosts. Off by default because the game refuses them (see `docs/poc-findings.md` finding 18); turn it on for a byte-identical candidate set to the simulator agent. |
| `--external-host IP` | Declare a host as outside the range. Without it the encoder falls back to `is_private()`, which is the right test in the simulator (public IPs exist there) and useless in a container range. |
| `--seed`, `--episodes`, `--metrics` | As expected; `--metrics` writes the stats as JSON. |

### What to expect, and what not to

The checkpoint in `models/` was cloned from **21 pairs of real container
behaviour covering three of five action types**, so playing it against a
simulator scenario measures transfer across a distribution gap in both
directions at once — different topology, different scale, and two action types
it has never emitted. A low win rate is the expected outcome and is
informative; treat it as a baseline for the loop, not as the surrogate's ceiling.

Two structural limits worth knowing before reading the numbers:

- **Attacker role only.** The policy has no `BlockIP` head, so it cannot play
  `AgentRole.Defender`. Training a defender *against* the surrogate works;
  running the surrogate as one does not.
- **No learning in this path.** `play` evaluates; it does not update weights.
  `SurrogateController.state_value()` exposes `V(s)` from the shared backbone for
  anyone wiring the surrogate into an RL loop later.

## Where things are stored

| Path | Contents | In git |
|---|---|---|
| `datasets/<run_id>/` | `pairs.jsonl`, `graphs/`, `unmappable.jsonl`, `report.json` | no |
| `models/<name>.pth` | checkpoint (~680 KB) | no |
| `models/<name>.metrics.json` | load report, per-epoch losses, eval metrics | no |

Both directories are git-ignored and fully regenerable from the observation
runs: a dataset copies every state graph it references, and those are ~2 MB
apiece (27 MB for the two runs here). Override the locations with
`NSG_SURROGATE_DATASETS` / `NSG_SURROGATE_MODELS`, or per-invocation with
`--out`.

## Module map

| Module | Needs torch | Responsibility |
|---|---|---|
| `state_adapter.py` | no | `graph.json` -> `GameState`, plus an `AdapterReport` of every dropped entity and forced choice, and a `Provenance` side channel (docker node ids, real service ports, external hosts) |
| `candidates.py` | no | valid-action enumeration via `netsecgame`'s own `generate_valid_actions`, plus one documented correction (no exfiltration from uncontrolled hosts) and exploit canonicalization |
| `state_diff.py` | no | NSG-level diff between two projected states; each category maps to exactly one action type |
| `labeling.py` | no | effect-primary + command-corroborating labels, the candidate gate, and the inventory of what NSG cannot express |
| `dataset.py` | no | trajectory walker, deduplication, and the self-contained dataset writer |
| `action_schema.py` | no | which parameter each factored head decides, per action type |
| `attempt_counts.py` | no | per-episode attempt counters, encoded into node features |
| `encoder.py` | yes | `GameState` -> `HeteroData`, feature-compatible with `sgrl_netsec` |
| `policy.py` | yes | `FactoredGNNPolicy` (parameter-identical to the simulator agent) and `SurrogatePolicy` |
| `nsg_agent.py` | yes | `SurrogateController` (socket-free decisions) and `SurrogateAgent` (a `BaseAgent` that plays episodes against the game server) |

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

Verified end to end on the real observation runs: state graph -> `GameState` ->
hetero graph -> four factored heads -> NSG action JSON, then labelled pairs ->
behaviour cloning -> a checkpoint that decodes actions again. 112 tests.

Not yet done:

- **`play` works against a live server** (verified on port 9099, 40 episodes
  across two configurations) but the surrogate cannot win any scenario yet: see
  the first-live-run table above.
- **21 training pairs, three of five action types.** The checkpoint is a
  pipeline-correctness artifact, not a usable policy. `ExploitService` and
  `ExfiltrateData` have no examples at all.
- **Only `strategic`-level runs are properly exercised.** `manual-run-strategic`
  is the one frozen run at the intended level; everything else was measured on
  an `operational` graph.
