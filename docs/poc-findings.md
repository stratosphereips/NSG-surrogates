# PoC findings: docker state graph -> NetSecGame action

What broke, what had to be decided arbitrarily, and what cannot be expressed at
all when a real container's state graph is projected into NetSecGame's state and
action vocabulary.

Evidence throughout is the one real observation run available on this machine,
`/opt/Agents/NSG-docker-state-creator/observation/manual-run` (311 nodes, 738
edges, `operational` level, schema `nsg-state-graph/1.1`), reproducible with:

```bash
python -m nsg_surrogate inspect /opt/Agents/NSG-docker-state-creator/observation/manual-run
```

Findings are ordered by how much they would hurt if left unaddressed.

---

## 1. Without a declared scope, the agent is pointed at the public internet

The state creator faithfully records every host the container contacted. In the
sample run that is 16 of 20 projected hosts: Ubuntu archive mirrors
(`91.189.9x.x`, `185.125.190.8x`), Cloudflare (`104.20.23.154`,
`172.66.147.243`), public DNS (`8.8.8.8`, `1.1.1.1`), and university resolvers
(`147.32.82.7`, `147.32.82.19`). 14 of them carry observed services.

NSG semantics treat every known host as attackable, so all 14 become valid
`ExploitService` targets. A randomly initialised policy driven through
`SurrogatePolicyAdapter` emitted, in its first five steps,
`ExploitService target_host=185.125.190.81` and
`ExploitService target_host=172.66.147.243` — Canonical and Cloudflare
infrastructure.

**Handled:** `AdapterConfig.scope_cidrs` / `--scope CIDR` drops out-of-scope
hosts (and their services) before projection; with `--scope 172.23.0.0/16` the
sample run projects 4 hosts and 18 candidate actions instead of 20 and 80. When
no scope is set the report says so explicitly.

**Still open:** the scope is operator configuration, not something the
observation carries. Nothing in the state graph distinguishes "range host" from
"host we fetched a package from", so this cannot be inferred — it has to be
declared per deployment, and any live run without it is unsafe.

## 2. Exfiltration — and therefore every NSG win condition — is unreachable

The state creator can report exactly one controlled host: the container it is
observing (`host:local`, via the `CONTROLS` edge). NSG's `ExfiltrateData`
requires a destination controlled host *different* from the data's host, so with
one controlled host the action space contains zero exfiltration actions. Every
NSG scenario's goal is defined in terms of exfiltrated data, so no episode can
ever be won in the real range.

**Handled:** `AdapterConfig.external_hosts` / `--external-host IP` designates
the attacker's own drop box, adds it to `controlled_hosts`, and reports the
addition as configuration rather than observation. With one external host the
sample run gains 4 `ExfiltrateData` candidates.

**Still open:** this is a declared fact injected into the state. If the drop box
is not reachable from the container, the translator will fail the action while
NSG's state says it should succeed.

## 3. Public/private is meaningless in a container range

Three simulator features are derived from `netaddr.IPAddress(...).is_private()`:
the host `is_public` bit, "data has been exfiltrated" (defined as *data present
on a controlled public host*), and the origin-is-public data bit. A docker range
is entirely RFC1918, so all three read 0 forever — the exfiltration destination
head has nothing to discriminate on, and the "already exfiltrated" signal never
fires.

**Handled:** the encoder's external test is driven by `Provenance.external_hosts`
(declared, per finding 2) and falls back to `is_private()` only when no external
host is declared.

## 4. The container's identity is ambiguous, and the naive choice is wrong

`host:local` claims 14 addresses, 4 of them routable IPv4:
`172.23.0.2, 172.17.0.2, 172.17.0.1, 172.23.0.1`. NSG identifies a host by a
single IPv4 string. Taking the numerically smallest — which is what a plain
`sorted()` does — picks `172.17.0.1`, a **docker bridge gateway** that is a
different machine and is separately present in the graph as its own host node.
Every action's `source_host` would then be attributed to the wrong host.

**Handled:** `_canonical_ipv4` ranks candidates: addresses inside a network the
host is `MEMBER_OF` first, then addresses no other host node owns, then
numerically. That selects `172.23.0.2`, the container's real address.

**Still open:** the aliasing itself is inexpressible. `172.23.0.1` and
`172.17.0.1` remain both "an address of the container" and "a separate known
host", so the surrogate can legitimately emit an action targeting its own host
under another identity.

## 5. Most of the graph has no NetSecGame counterpart

| Node type | Count | Fate |
|---|---|---|
| `data` | 136 | projected, heavily filtered (finding 6) |
| `program` | 64 | **no representation** |
| `host` | 30 | projected |
| `command` | 28 | **no representation** |
| `service` | 27 | projected |
| `network` | 14 | projected, filtered |
| `block` | 7 | **not projectable** (finding 8) |
| `user` | 3 | **no representation** |

Of 26 edge types, NSG can express roughly four relationships. `program`,
`command` and `user` are 95 of 311 nodes — and users and credentials are the
actual pivot material in a real range. The `strategic` detail level drops
process detail by design, so this loss is intended rather than accidental, but
it bounds what any NSG-shaped agent can reason about.

## 6. Data nodes are mostly OS noise, and NSG has no way to tell

136 data nodes, of which:

- 71 are container/OS noise paths (`/tmp/apt.sig.*`, `/var/lib/apt/...`);
- 58 have `exists: [true, false]` — observed both present and absent, i.e.
  created and then deleted;
- only 6 carry real file metadata (`sha256`, `size`, `mode`);
- `existence` is present on just 20 nodes, and its only value there is
  `"unknown"`.

Projected raw, `known_data` becomes garbage, and since the exfiltration action
space is `|data| x |controlled hosts|`, it inflates the branching factor with
items no attacker cares about.

**Handled:** confidence threshold, a path denylist, a per-host cap, and a
presence test that accepts only an unambiguous `exists: true`. The sample run
projects 4 data items instead of 136.

**Still open:** "important data" is a judgement the state creator makes at
`strategic` level with its own filter; the two filters overlap and should
probably be reconciled rather than stacked.

## 7. Confidence and inference are discarded

Nodes carry `confidence` (0.35–1.0), and inferred entities carry
`inferred=true`, `inference_method`, `inferred_from`. NSG knowledge is boolean:
a fact is known or not. Every projection therefore either promotes a hypothesis
to certainty or discards it.

Concretely, 11 of 14 networks in the sample run are **assumed /24s** around
observed public IPs (`8.8.8.0/24`, `147.32.82.0/24`), at confidence 0.35. Keeping
them would let `ScanNetwork` sweep third-party internet ranges; they are dropped
by default, which also means the agent cannot scan anything it has not already
been told about.

Nothing currently carries uncertainty into the policy. The node feature vectors
have 11 unused slots (features 5–15), so a confidence channel is available
whenever it is wanted — at the cost of diverging from simulator-trained weights.

## 8. Firewall blocks cannot be projected at all

7 `block` nodes exist, each attributed to a *target service* with
`hypothesis: true`, `reason: "network-no-reply"`, confidence 0.78. NSG's
`known_blocks` is `Dict[IP, Set[IP]]` — keyed by the source host that observed
the block. The state creator's schema has no source attribution, so guessing one
would fabricate a fact. `nsg-action-translator`'s projection reaches the same
conclusion independently.

Consequence: `generate_valid_actions` cannot prune blocked actions, so the
surrogate will keep proposing actions the real range silently drops.

## 9. Only two of five action types can actually be executed

Per `nsg-action-translator`'s capability table: `ScanNetwork` and `FindServices`
are live; `FindData` is unsupported; `ExploitService` and `ExfiltrateData` are
blocked. Measured on the sample run, **56% of candidate actions are executable**
(10 of 18 with a scope declared, 44 of 80 without).

So an NSG-trained policy can currently execute its discovery prefix and nothing
past it, and a policy that learned to exploit early will spend most of its steps
emitting actions that never start a process. The adapter tracks this:
`SurrogatePolicyAdapter.info()["emitted_unexecutable"]`.

## 10. Two simulator features are broken, one of them silently

- **`service[0]` (port) was always zero.** `sgrl_netsec/policy_netsec.py:115`
  computes `int(service.name.split("/")[0])`, but `Service.name` holds a service
  *name* (`ssh`, `https`), so the parse raises `ValueError`, the exception is
  swallowed, and the feature is dead in the trained policy. The state creator
  reports real ports, which the encoder now uses via `Provenance.service_ports`.
  Because a simulator-trained checkpoint has never seen a nonzero value here,
  `legacy_service_port=True` reproduces the dead feature for A/B comparison.
- **`state_summary[1]` reads 0 in the real range.** The "data not yet
  exfiltrated" term counts data on hosts the agent does *not* control. In the
  container every data item is found on the local (controlled) host, so the
  phase-of-attack signal that should say "time to exfiltrate" is structurally
  silent.

Also stale: `docs/blackbox_pure_gnn_arch.md` still documents services as
deduplicated across hosts, while `state_to_pyg` now creates one node per
`(host, service)` pair.

## 11. NSG's action generator forces choices the real world does not

- `FindData` is generated with `source_host == target_host`, so it is only ever
  "look for data on a host I already control" — one candidate per controlled
  host. In a real container, finding data is most of what an operator does.
- `ExploitService` is canonicalized to the lexicographically smallest valid
  service per (source, target) pair, because in the simulator the service choice
  does not change the outcome. In a real range the service *is* the exploit.
  Kept for weight compatibility, flagged in `candidates.canonicalize_exploits`.

## 12. IPv6 and DNS-named hosts vanish

10 of 30 hosts have no routable IPv4 address (IPv6-only endpoints and
DNS-name-identified hosts). NSG's `IP` is an IPv4 string, so they are dropped
rather than guessed, and any service attached to them goes with them (1 service
in the sample run).

## 13. State is not step-synchronous, and there is no reward or episode end

The state graph is rebuilt on a timer (`OBS_STATE_INTERVAL`, default 10s), and
trajectory actions are batched by a settle window. So an action's effect is not
guaranteed to appear in the next state read, there is no per-step reward, and
nothing signals episode termination. The attempt counters in `attempt_counts.py`
are the only reliable per-step memory the surrogate has.

## 14. The reverse direction has almost no supervision in this corpus

For the eventual docker-action -> NSG-action mapping: of 28,864 trajectory
records in the sample run, **28,807 actions are classified `generic_command`**.
Only 18 are classified at all (9 `network_probe`, 5 `data_access`,
3 `network_request`, 1 `package_change`), and those are `ping 1.1.1.1` and
`find .` — neither of which has a clean NSG equivalent. Action records do carry
good raw material (`command.argv`, `shell_text`, `changed_domains`,
`exit_status`, a `delta` pointer, `schema_version: nsg-action/1.0`), but real
attack traces from `stratocyberlab` challenges will be needed before that
mapping can be fit.

## 15. Scale differs from the training distribution

The sample run has 30 hosts, 27 services and 136 data items against the
simulator's tiny topologies (a handful of each). Attempt counters and the state
summary both clip at 10 before normalising, so they saturate. A policy trained on
`two_networks_tiny` is out of distribution here even after the mapping is
correct.

## 16. Label yield from a real trajectory is very low

The trajectory layer already pairs states with actions — every action record
names `state_before` / `state_after`, and each `trajectory/states/state-NNNNNN/`
holds a full graph. Measured with `dataset` on `manual-run`:

| | `--scope 172.23.0.0/16` | no scope |
|---|---:|---:|
| states / transitions | 39 / 38 | 39 / 38 |
| action records | 36,164 | 36,164 |
| after deduplication | 752 | 752 |
| **labels** | **8** | **13** |
| transitions with no label | 30 | 26 |
| NSG-visible diff empty | 32 | 22 |
| off-vocabulary commands | 315 | 228 |

Attribution quality explains most of it: 36,123 of the 36,164 records come from
`ebpf-exec-fallback` (confidence 0.80), 28 from `bash-command` (0.995) and 13
from process lifecycle, and the deltas that exist say `causal_isolation: false`
with dozens of `contributing_action_ids` — the state change belongs to a batch,
not to an action. Scoping cuts the yield further because most observed activity
(package fetches) targets hosts outside the range.

So this corpus supports the *fidelity measurement* it was built for, not
training. Real attack traces from `stratocyberlab`, with `bash-command`
attribution and a declared scope, are the corpus that would.

## 17. What NetSecGame cannot express (extension candidates)

Every unlabelable transition is categorised, and each category names a
capability NSG would need. From `manual-run`, unscoped:

| Occurrences | Category | What it means |
|---:|---|---|
| 228 | command outside the NSG action vocabulary | `apt`, `pip`, `ip`, `ss`, editors — no action type covers them |
| 16 | host discovered outside any known network | a host became known via DNS resolution or an outbound connection; NSG can only learn hosts by scanning a network it already knows |
| 15 | services learned on a previously unknown host | host and its services appeared in one transition; NSG needs `ScanNetwork` then `FindServices` as two steps |
| 1 | network learned without a scan | NSG has no action that discovers a network — they come from starting knowledge |
| 1 | control gained without prior service knowledge | credential reuse, key-based login and lateral movement have no representation: `ExploitService` requires a known service |

Three further categories the labeler emits, not triggered by this run but
demonstrated in `tests/test_labeling.py`:

- **blind exfiltration of undiscovered data** — `scp /etc/shadow drop:/` from a
  controlled host. Controlling a host should be enough to take a file from it,
  but NSG enumerates exfiltration over `known_data` only, so an operator who
  skips `FindData` performs an action the vocabulary cannot express. This is the
  clearest extension candidate: it needs an exfiltration action parameterized by
  a path rather than a discovered `Data` node.
- **data learned on an uncontrolled host** — reading a remote page, an open
  share or a service banner. `FindData` only enumerates data on controlled hosts.
- **data moved from an uncontrolled host** — the access that made it possible is
  unrepresented (see also finding 18).

Also unrepresentable in principle: **NSG knowledge is monotonic**. The real graph
retracts facts (a file is deleted, a host stops answering, confidence decays) —
`manual-run` loses 7 services, 1 network, 1 host and 1 controlled host across
its transitions — and no NSG action can shrink a state, so those transitions
cannot be labelled at all.

## 18. NetSecGame's action generator emits exfiltrations the game refuses

`generate_valid_actions` (`netsecgame/utils/utils.py:295`) builds
`ExfiltrateData` by iterating `state.known_data` for the source host and never
checks that the agent controls it:

```python
for source_host, data_list in state.known_data.items():
    for data in data_list:
        for trg_host in state.controlled_hosts:
```

Knowing that data sits on a host is not being able to take it, so the generated
action space is larger than the game's semantics — and any agent trained against
it has probability mass on actions that can only fail. The same shape is
duplicated in `nsg-action-translator/nsg_action_translator/policies/valid_actions.py`.

`candidates.enumerate_actions` corrects this by default
(`require_controlled_exfil_source=True`) and can reproduce the raw generator for
comparison. Worth fixing upstream, since it also affects the simulator agents.

Note this is the opposite direction from finding 17's blind exfiltration: the
generator is simultaneously too permissive about the *source host* and too
restrictive about *which data* can be taken.

---

## What works

- **Checkpoint transfer is structurally sound.** `tests/test_policy.py` builds
  this repo's `FactoredGNNPolicy` and `sgrl_netsec`'s side by side, forwards both
  (GATv2 is lazily initialised, so parameters only exist after a forward), and
  asserts identical `state_dict` keys and shapes plus a successful
  `load_state_dict`. Both tests pass, so a simulator-trained checkpoint will
  load and mean the same thing here. No trained `.pth` exists on this machine,
  so behaviour after loading is untested.
- **The full path runs on real observation data.** `inspect`, `state` and `act`
  all work against `observation/manual-run`; `act` emits an `Action` whose JSON
  the translator accepts, with all four heads firing and their choices recorded.
- **The mapping is deterministic and tested.** 77 tests, of which the mapping
  and labelling halves need only the standard library.
- **The pairing comes for free.** The trajectory layer's `state_before` /
  `state_after` references plus the per-state graphs mean `dataset` needs no
  alignment heuristics; only the *labels* had to be derived.

## Recommended next steps

1. Rebuild an observation run at `--level strategic` and re-run `inspect`; the
   input contract this repo targets has only been exercised at `operational`.
2. Get a trained checkpoint onto this machine and measure, on real projected
   states, how often the policy emits an unexecutable action and whether its
   choices are sane at all.
3. Decide scope and drop-box declaration as deployment config (findings 1 and 2)
   — they are prerequisites for any live run, not tuning knobs.
4. Take the block-attribution gap (finding 8) to the state-creator side: adding
   an observing-source field would make `known_blocks` projectable.
5. Capture real attack traces in `stratocyberlab` before treating the trajectory
   corpus as training data (findings 14 and 16); the current run supports a
   fidelity measurement only.
6. Decide which of finding 17's categories justify extending NetSecGame. The
   strongest case is blind exfiltration (an action parameterized by a path
   rather than a discovered `Data` node), followed by host discovery outside a
   known network and control gained without prior service knowledge.
7. Report finding 18 upstream — the exfiltration source-host check is missing in
   both NetSecGame and the translator's copy of the generator.
