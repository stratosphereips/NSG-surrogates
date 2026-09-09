# Findings: mapping an emulated container's state to NetSecGame

This document records what happens when the state graph produced by
NSG-docker-state-creator inside a real container is converted into NetSecGame's
state and action vocabulary: which conversions are exact, which required a
choice that the observation does not determine, and which observed facts have no
NetSecGame equivalent at all.

Two observation runs provide the evidence:

| | `manual-run-strategic` | `manual-run` |
|---|---|---|
| detail level | `strategic` (the intended level) | `operational` |
| state graph | 2,048 nodes, 4,112 edges | 311 nodes, 738 edges |
| still being written | no | yes, so counts increase between builds |
| schema | `nsg-state-graph/1.1` | `nsg-state-graph/1.1` |

Both are reproducible with:

```bash
python -m nsg_surrogate inspect <observation-dir> --scope 172.23.0.0/16 --external-host 10.9.9.9
python -m nsg_surrogate dataset <observation-dir> --scope 172.23.0.0/16 --external-host 10.9.9.9
```

Findings are grouped by the part of the mapping they concern. Numbers are stable
so that other documents can cite them, which is why they are not consecutive
within a section. Each finding states the observation, then what this repository
does about it, then what remains unresolved.

| # | Finding | Section |
|---:|---|---|
| 1 | Every host the container contacted becomes an attack target | projection |
| 2 | Exfiltration is unreachable, so no scenario goal can be met | projection |
| 3 | The public/private distinction carries no information here | projection |
| 4 | A multi-homed container has no single NetSecGame identity | projection |
| 5 | Most of the state graph has no NetSecGame counterpart | projection |
| 6 | Data nodes are dominated by operating-system activity | projection |
| 7 | Confidence and inference markers are discarded | projection |
| 8 | Firewall blocks cannot be projected | projection |
| 9 | Downstream execution covers two of the five action types | action vocabulary |
| 10 | Two encoder features are inoperative | action vocabulary |
| 11 | NetSecGame's generator constrains two parameters that reality does not | action vocabulary |
| 12 | IPv6-only and DNS-named hosts are discarded | projection |
| 13 | Observations are not synchronised with actions | attribution |
| 14 | Most recorded actions are unclassified | attribution |
| 15 | The projected state is larger than the simulator's training topologies | projection |
| 16 | Label yield per session is low, and bounded by attribution quality | attribution |
| 17 | Observed behaviour with no NetSecGame representation | action vocabulary |
| 18 | NetSecGame's generator emits exfiltrations the game rejects | action vocabulary |
| 19 | What the strategic run changed, and what it revealed | attribution |
| 20 | Accuracy is bounded by how many states the projection can distinguish | fitting |
| 21 | A trajectory may involve more than one machine | attribution |

---

# A. Projecting the state

## 1. Every host the container contacted becomes an attack target

The state creator records all hosts the container communicated with. In
`manual-run`, 16 of the 20 projected hosts are public internet addresses: Ubuntu
archive mirrors (`91.189.9x.x`, `185.125.190.8x`), Cloudflare
(`104.20.23.154`, `172.66.147.243`), public resolvers (`8.8.8.8`, `1.1.1.1`),
and university resolvers (`147.32.82.7`, `147.32.82.19`). Fourteen of them have
observed services, and NetSecGame treats every known host with a known service
as a valid `ExploitService` target.

This is not hypothetical: a randomly initialised policy selected
`ExploitService` against `185.125.190.81` and `172.66.147.243` — Canonical and
Cloudflare infrastructure — within its first five steps.

**Implemented.** `AdapterConfig.scope_cidrs` (`--scope CIDR`) discards hosts
outside the declared networks, and their services with them. With
`--scope 172.23.0.0/16`, `manual-run` projects 4 hosts and 18 candidate actions
instead of 20 and 80. When no scope is given, the report states how many
projected hosts are public addresses.

**Unresolved.** The scope is deployment configuration. Nothing in the state
graph distinguishes a host that belongs to the range from a host that served a
package download, so the distinction cannot be inferred and must be declared.

## 2. Exfiltration is unreachable, so no scenario goal can be met

The state creator can report exactly one controlled host: the container it
observes, via the `CONTROLS` edge from `agent:observed`. NetSecGame's
`ExfiltrateData` requires a destination controlled host different from the one
holding the data, so with a single controlled host the candidate set contains no
exfiltration action. Since NetSecGame scenario goals are defined in terms of
exfiltrated data, no episode can be completed.

**Implemented.** `AdapterConfig.external_hosts` (`--external-host IP`) declares a
host outside the range, adds it to `controlled_hosts`, and records in the report
that the addition is configuration rather than observation. One such host gives
`manual-run` four exfiltration candidates.

**Unresolved.** The declared host is an assertion about the world that the
observation does not support. If it is not reachable from the container, the
projected state will imply that exfiltration is possible when it is not.

## 3. The public/private distinction carries no information here

Three encoder features derive from `netaddr.IPAddress(...).is_private()`: whether
a host is public, whether a data item has been exfiltrated (defined in the
simulator as present on a controlled public host), and whether a data item's
origin host is public. A container network is entirely RFC1918, so all three are
zero for every entity: the destination head has nothing to discriminate on, and
the exfiltration indicator never becomes true.

**Implemented.** The encoder's external-host test reads
`Provenance.external_hosts`, populated from the declaration in finding 2, and
falls back to `is_private()` only when no host has been declared — which is the
correct test inside the simulator, where public addresses exist.

## 4. A multi-homed container has no single NetSecGame identity

NetSecGame identifies a host by one IPv4 string. In `manual-run`, `host:local`
carries 14 addresses, 4 of them routable IPv4: `172.23.0.2`, `172.17.0.2`,
`172.17.0.1`, `172.23.0.1`. Selecting the numerically smallest, which is what
sorting produces, yields `172.17.0.1` — a Docker bridge gateway, which is a
different machine and appears separately in the graph as its own host node. Every
action's `source_host` would then name the wrong host.

**Implemented.** `_canonical_ipv4` ranks the candidate addresses: first those
inside a network the host is a `MEMBER_OF`, then those no other host node claims
as its only address, then numerically. This selects `172.23.0.2`, the
container's own address in its own subnet.

**Unresolved.** The aliasing itself cannot be represented. `172.23.0.1` and
`172.17.0.1` remain both addresses of the container and separate known hosts, so
the surrogate can select an action against its own host under another identity.

## 5. Most of the state graph has no NetSecGame counterpart

Node types in `manual-run`:

| Node type | Count | Outcome |
|---|---:|---|
| `data` | 136 | projected, after filtering (finding 6) |
| `program` | 64 | no representation |
| `host` | 30 | projected |
| `command` | 28 | no representation |
| `service` | 27 | projected |
| `network` | 14 | projected, after filtering |
| `block` | 7 | not projectable (finding 8) |
| `user` | 3 | no representation |

Of 26 edge types, NetSecGame can express approximately four relationships.
`program`, `command` and `user` account for 95 of the 311 nodes. Users and
credentials are the primary means of lateral movement in a real range, and they
have no representation in the target vocabulary. The `strategic` detail level
omits process detail by design, so part of this loss is intended, but it bounds
what any NetSecGame-shaped agent can reason about.

## 6. Data nodes are dominated by operating-system activity

Of the 136 data nodes in `manual-run`:

- 71 are container or OS paths (`/tmp/apt.sig.*`, `/var/lib/apt/...`);
- 58 have `exists: [true, false]`, meaning the file was observed both present
  and absent, so it was created and then deleted;
- 6 carry file metadata (`sha256`, `size`, `mode`);
- `existence` is present on 20 nodes, and its only value there is `"unknown"`.

Projected without filtering, `known_data` consists mostly of installer
temporaries. The exfiltration candidate set has size |data| × |controlled
hosts|, so this also inflates the action space.

**Implemented.** A confidence threshold, a path denylist, a per-host cap, and a
presence test. The presence test accepts `exists: true`, and also accepts a file
named on the operator's command line even when the filesystem monitor never
confirmed it — see finding 19, where that rule decides whether `/etc/passwd` is
projected. `manual-run` then projects 4 data items instead of 136.

**Unresolved.** The state creator applies its own "important files" filter at the
`strategic` level. The two filters overlap and should be reconciled rather than
composed.

## 7. Confidence and inference markers are discarded

Nodes carry `confidence` between 0.35 and 1.0, and inferred entities carry
`inferred`, `inference_method` and `inferred_from`. NetSecGame knowledge is
boolean: a fact is known or it is not. Every projection therefore either
promotes an uncertain statement to certainty or discards it.

Concretely, 11 of the 14 networks in `manual-run` are `/24` prefixes assumed
around observed public addresses (`8.8.8.0/24`, `147.32.82.0/24`) at confidence
0.35. Retaining them would allow `ScanNetwork` to target third-party address
ranges; they are discarded by default, which also means the agent cannot scan
anything it has not already been told about.

No uncertainty is currently passed to the policy. The node feature vectors have
11 unused positions (indices 5 to 15), so a confidence channel is available.
Compatibility with the simulator agent's feature layout is not maintained, so
using them is a free choice; the positions are empty because no measurement yet
shows what to put there.

## 8. Firewall blocks cannot be projected

`manual-run` contains 7 `block` nodes; the strategic run contains 1,001. Each is
attributed to a target *service*, with `hypothesis: true`,
`reason: "network-no-reply"` and confidence 0.78. NetSecGame's `known_blocks` is
`Dict[IP, Set[IP]]`, keyed by the host that observed the block. The state
creator's schema has no such attribution, so any projection would have to invent
the source host. `nsg-action-translator`'s projection reaches the same
conclusion independently.

Consequence: `generate_valid_actions` cannot exclude blocked actions, so the
surrogate will continue to propose actions that the range silently drops.

**Request to the state creator.** A field naming the host that observed the
failed connection would make this category projectable.

## 12. IPv6-only and DNS-named hosts are discarded

Ten of the 30 hosts in `manual-run` have no routable IPv4 address: IPv6-only
endpoints and hosts identified only by DNS name. NetSecGame's `IP` is an IPv4
string, so these hosts are discarded rather than approximated, and services
attached to them are discarded with them (one service in `manual-run`).

## 15. The projected state is larger than the simulator's training topologies

`manual-run` projects 30 hosts, 27 services and 136 data items before filtering.
The simulator scenarios used to train the reference agents contain a handful of
each. The attempt counters and the phase-of-attack summary both saturate at 10
before normalisation, so on a state of this size they are constant. A policy
trained on `two_networks_tiny` is therefore outside its training distribution
here even when the mapping is correct.

---

# B. Attributing actions to state changes

## 13. Observations are not synchronised with actions

The state graph is rebuilt on a timer (`OBS_STATE_INTERVAL`, 10 seconds by
default), and recorded actions are grouped by a settle window. An action's
effect is therefore not guaranteed to appear in the next state snapshot. There is
also no per-step reward and no episode termination signal. The attempt counters
in `attempt_counts.py` are the only per-step history available to the surrogate.

## 14. Most recorded actions are unclassified

Of 28,864 trajectory records in an early build of `manual-run`, 28,807 were
classified `generic_command`. Only 18 carried a specific classification: 9
`network_probe`, 5 `data_access`, 3 `network_request`, 1 `package_change`. The
classified ones were `ping 1.1.1.1` and `find .`, neither of which has an exact
NetSecGame equivalent.

The records do contain the information needed for labelling — `command.argv`,
`shell_text`, `changed_domains`, `exit_status`, a pointer to the state delta,
and `schema_version: nsg-action/1.0` — so the limitation is the content of the
recorded session, not the record format.

## 16. Label yield per session is low, and bounded by attribution quality

The trajectory layer already pairs states with actions: every action record
names `state_before` and `state_after`, and each `trajectory/states/state-*/`
directory holds a complete state graph. Labelling therefore needs no alignment
heuristics. The yield is nevertheless small:

| | `manual-run-strategic` | `manual-run` |
|---|---:|---:|
| states / transitions | 12 / 22 | 39 / 65 |
| action records | 807 | 43,202 |
| records after deduplication | 271 | 752 |
| labels | 8 | 13 |
| transitions with no label | 14 | 52 |
| transitions with no change in the projected state | 20 | 57 |
| commands with no NetSecGame equivalent | 98 | 537 |

Two properties of the recordings explain most of this.

**Collector attribution.** In the strategic run, 783 of 807 records come from
the eBPF exec fallback (confidence 0.80), 19 from the interactive shell
(confidence 0.995) and 5 from process lifecycle. The high-confidence source is
the one that reliably identifies the operator's own command.

**Scope reduces yield.** Built without a scope, the same runs yield 8 and 24
labels rather than 8 and 13: most of the additional labels describe discovery of
out-of-range hosts, which is activity the surrogate should not learn to imitate.
The reduction is intended.

**No causal isolation.** Every state delta in both runs reports
`causal_isolation: false`, with dozens of entries in
`contributing_action_ids`. No state change in this corpus can be attributed to a
single command. Effect-derived labels are therefore batch-level, which bounds
their confidence, and the same command can be labelled under more than one
transition (finding 19).

## 21. A trajectory may involve more than one machine

A recording is produced by an observer inside one container, so the commands it
captures were executed there. It does not follow that the trajectory concerns a
single machine: an agent that gains control of a second host acts from there as
well, and every NetSecGame action carries a `source_host`.

The recorded evidence does not identify the issuing host. An action record
carries `actor` (pid, tty, uid, and a username on 5 of 807 records in the
strategic run), `targets` (the addresses the action was aimed at) and `scope`
(`unknown` on 796 records, `network` on 6, `local` on 4). None of these names the
host from which the command was issued, because the observer only ever sees its
own container.

Two ways a second machine enters a trajectory, attributed differently:

* **a second observer on that host.** Its own container is `host:local` in its
  own recording, so attribution is correct per recording. Combining recordings
  then requires each run's local host, which the projection already reports in
  `Provenance.local_hosts`.
* **remote execution from the observed host**, such as `ssh host "nmap ..."`.
  The observer records the wrapper but not what it ran, so the inner action's
  source host cannot be recovered from this evidence.

**Implemented.** `infer_source_host` prefers the observed host, which is correct
for commands this observer recorded, but records the choice as a note whenever
the state contains more than one controlled host — the recording cannot
establish that the observed host was the one that acted. A remote-execution
wrapper naming an already-controlled host is reported as
`command executed on another host` rather than attributed to the observed host.
A wrapper naming a host that is not yet controlled is an access attempt, and is
labelled `ExploitService` by the ordinary rules.

**Unresolved.** For actions taken through a wrapper, neither the action nor the
state it changed on the remote host is in this recording. Recovering them
requires an observer on that host.

## 19. What the strategic run changed, and what it revealed

`manual-run-strategic` is the first run at the detail level this repository
targets, and it is not being written to, so builds are reproducible. It is
substantially better suited to labelling than `manual-run`:

| | `manual-run` | `manual-run-strategic` |
|---|---:|---:|
| action records | 43,202 | 807 |
| interactive-shell records | 28 | 19 |
| states | 39 | 12 |
| labels | 13 | 8 |
| action types covered | 2 | 3 |

The session is also attack-shaped: `ping`, `apt install nmap`,
`nmap -sP -n 172.23.0.0/24`, `nmap -sS 172.23.0.1`, `cat /etc/passwd`,
`ssh test@172.23.0.1 -p 902`. It exposed four problems, all now handled.

**Port-scan results are attributed to the scanning host.** `nmap -sS` against a
single target produced 999 service nodes with `status: "attempted"` (confidence
0.7, `zeek_state: S0`) and 1,000 `network-no-reply` block hypotheses. Only 5 of
the 1,007 service nodes carry `open` or `listening`. The 999 are attributed to
`host:local`, the host that performed the scan, rather than to the host that was
scanned. Projected as-is, they state that the agent runs 999 services on itself.
`AdapterConfig.service_status_denylist` now discards nodes whose only status is
`attempted` or `targeted`, retaining any node that also carries positive
evidence.

The mis-attribution is the more important half, and it belongs to the state
creator: the scanned ports describe the target. Until it is corrected, the result
of a service scan — which is exactly what `FindServices` is defined to produce —
is not observable in the projected state.

**A file discovered on the command line was being discarded.** `/etc/passwd`
appears with `knowledge_source: "command-argument"` and
`existence: "unknown"`, sourced from the TTY log, because the filesystem monitor
never reconciled `/etc`. Requiring filesystem confirmation retained two debconf
caches and discarded the file the operator actually read. In NetSecGame,
`known_data` means the agent has discovered the data, and a file named on the
command line establishes that, so `_data_is_present` now accepts it
(`accept_command_argument_data`).

**Actions with no effect were being skipped.** An action that changes nothing is
recorded with `state_before == state_after`; `nmap -sP` appears as
`state-000005 -> state-000005`. Enumerating transitions as consecutive pairs of
state identifiers never matches those records. Transitions are now taken from
the action records themselves, which also recovers multi-snapshot transitions
such as `state-000003 -> state-000005`. This raised the run from 4 labels to 8.
These are also the transitions the dataset most needs: a dataset of only
state-changing transitions represents every action as successful.

**Operators do not scan the networks NetSecGame knows about.** The container
knows `172.23.0.0/16`; the operator scanned `172.23.0.0/24`. `ScanNetwork` takes
a known `Network` object, so the label uses the containing network and records
the difference in granularity in the label's notes. A scanned range that no
known network contains is reported as unmappable.

Two properties remain, both relevant to the next capture:

- **A successful service scan can leave no trace in the projected state.**
  `nmap -sS 172.23.0.1` advanced the recorded state, but the projection is
  unchanged once the misattributed artifacts are filtered, so the label falls
  back to command-only evidence. Labels now distinguish the two cases: "docker
  state advanced but its NSG projection is unchanged" versus "no state change
  followed this command".
- **Effect-only labels can originate from observer artifacts.** One
  `FindServices` label derives from services appearing during `apt update`
  rather than from a scan. No command in that batch matched the implied action
  type, which is why the label is recorded as effect-only at confidence 0.54.

---

# C. The action vocabulary

## 9. Downstream execution covers two of the five action types

Recorded for context; this belongs to `nsg-action-translator` rather than to this
repository. Its capability table lists `ScanNetwork` and `FindServices` as
executable, `FindData` as unsupported, and `ExploitService` and `ExfiltrateData`
as blocked. A policy trained on NetSecGame can therefore execute its discovery
prefix in the emulated range and nothing beyond it.

## 10. Two encoder features are inoperative

- **Service port, feature index 0, is always zero in the simulator agent.**
  `sgrl_netsec/policy_netsec.py:115` computes `int(service.name.split("/")[0])`,
  but `Service.name` holds a service name such as `ssh` or `https`. The
  conversion raises `ValueError`, the exception is caught, and the feature
  remains zero for every service, so the trained policy has never received a
  non-zero value there. The state creator reports the real port number, which
  this repository's encoder uses via `Provenance.service_ports`. Passing
  `legacy_service_port=True` reproduces the original behaviour for comparison.
- **The second phase-of-attack feature is always zero in a container.** It counts
  data items on hosts the agent does *not* control. In the emulated range every
  data item is found on the local, controlled host, so the feature that should
  indicate "data is available to exfiltrate" is constant.

Separately, `sgrl_netsec/docs/blackbox_pure_gnn_arch.md` states that service
nodes are deduplicated across hosts; `state_to_pyg` creates one node per
`(host, service)` pair.

## 11. NetSecGame's generator constrains two parameters that reality does not

- `FindData` is generated with `source_host == target_host`, so it can only
  express searching a host the agent already controls: one candidate per
  controlled host. Searching for data is a large fraction of what an operator
  actually does.
- `ExploitService` is reduced to the lexicographically smallest valid service
  per (source, target) pair, because in the simulator the choice of service does
  not affect the outcome. In a real range the service determines whether an
  exploit exists. The reduction is retained because the recorded labels cannot
  yet identify which service was used, and is documented in
  `candidates.canonicalize_exploits`.

## 17. Observed behaviour with no NetSecGame representation

Every transition that cannot be labelled is categorised, and each category names
a capability NetSecGame would need. Counts below are from builds without a scope,
so that out-of-range activity is included; `manual-run` is still being written,
so its counts increase between builds.

| Category | `manual-run-strategic` | `manual-run` | Explanation |
|---|---:|---:|---|
| command outside the action vocabulary | 74 | 402 | `apt`, `pip`, `ip`, `ss`, editors: no action type covers them |
| host discovered outside any known network | 19 | 20 | the host became known through DNS resolution or an outbound connection; NetSecGame can only learn hosts by scanning a network it already knows |
| services discovered on a previously unknown host | 3 | 19 | host and services appeared in one transition; NetSecGame requires `ScanNetwork` and then `FindServices` |
| network learned without a scan | 0 | 2 | NetSecGame has no action that discovers a network; networks come from initial knowledge |
| control gained without prior service knowledge | 0 | 1 | credential reuse, key-based login and lateral movement have no representation, because `ExploitService` requires a known service |
| data discovered on an uncontrolled host | 0 | 1 | `FindData` enumerates only data on controlled hosts |

Three further categories are implemented and demonstrated in
`tests/test_labeling.py`, though this corpus does not trigger them:

- **exfiltration of undiscovered data.** `scp /etc/shadow drop:/` from a
  controlled host. Controlling a host is sufficient to copy a file from it, but
  NetSecGame enumerates exfiltration over `known_data`, so an operator who does
  not run `FindData` first performs an action the vocabulary cannot express.
  This is the clearest candidate for an extension: an exfiltration action
  parameterised by a path rather than by a discovered `Data` object.
- **data discovered on an uncontrolled host**, such as reading a web page, an
  open share, or a service banner. `FindData` enumerates only data on controlled
  hosts.
- **data copied from an uncontrolled host**: the access that made the copy
  possible has no representation.

A further property is structural rather than a missing action: **NetSecGame
knowledge is monotonic**, while the recorded graph retracts facts. A file is
deleted, a host stops responding, or confidence decays below the threshold.
Measured without a scope, `manual-run` loses 5 services, 2 data items, 1
network, 1 host and 1 controlled host across its transitions; the strategic run
loses none. No NetSecGame action reduces a state, so these transitions cannot be
labelled.

With a scope declared, the retractions disappear from the measurement, because
the entities being retracted are outside the range and are filtered before the
comparison. The retraction has not stopped happening; it is no longer visible.

## 18. NetSecGame's generator emits exfiltrations the game rejects

`generate_valid_actions` (`netsecgame/utils/utils.py:295`) constructs
`ExfiltrateData` by iterating `state.known_data` for the source host, without
checking that the agent controls that host:

```python
for source_host, data_list in state.known_data.items():
    for data in data_list:
        for trg_host in state.controlled_hosts:
```

Knowing that data resides on a host does not imply being able to copy it from
there. The generated action space is therefore larger than the game's semantics,
and any agent trained against it assigns probability to actions that can only
fail. The same construction appears in
`nsg-action-translator/nsg_action_translator/policies/valid_actions.py`.

`candidates.enumerate_actions` excludes these actions by default
(`require_controlled_exfil_source=True`) and can enumerate exactly what the
installed `netsecgame` produces for comparison.

**Status: reported upstream and accepted for fixing.** Once the control check is
added to `generate_valid_actions`, the filter here removes nothing and the
default remains correct. `tests/test_labeling.py` is written so that both states
of the upstream code pass: one test asserts the invariant that this
repository's candidate set never contains such an action, and a second records
whether the upstream generator still produces one, without asserting that it
does.

Note the two directions: the generator is simultaneously too permissive about
the source host and too restrictive about which data may be copied (finding 17).

---

# D. Fitting a policy

## 20. Accuracy is bounded by how many states the projection can distinguish

Fitting the four heads to the 21 available pairs reached 0.7143 accuracy for
every random seed. That value is the identifiability limit of the data rather
than a property of the model: without interaction history the 21 samples reduce
to **8 distinct encoder inputs**, and 6 of those carry more than one distinct
label, because the same projected state was labelled `FindData`, `FindServices`
and `ScanNetwork` at different points in the session.

Replaying each run in recorded order and attaching the accumulated
`AttemptCounts` to each sample makes all 21 inputs distinct, raises the limit to
1.00, and the model then reaches it exactly.

The consequence for data collection is that sample count is not the binding
constraint. Additional sessions of the same activity produce rows whose encoder
inputs collide with existing ones. What increases the limit is either activity
that changes the projected state (exploiting a host, gaining control, copying
data) or a projection that preserves more of the observed distinctions —
finding 7 notes that 11 feature positions are unused.

---

# Verified

- **The projection and labelling path runs on real observation data.**
  `inspect`, `state`, `dataset` and `act` all operate on both runs.
- **The agent runs against a live NetSecGame server.** Twenty episodes with the
  fitted checkpoint and twenty with a randomly initialised policy; the fitted
  policy's action distribution follows the training distribution, and it selects
  `ExploitService` zero times where the untrained policy selects it 78 times.
  Neither wins, because the training set contains no exploitation or
  exfiltration (README, *Measured behaviour of the current checkpoint*).
- **Determinism and coverage.** 114 tests; the projection and labelling tests
  require only the standard library, and no test requires another repository to
  be checked out.
- **Pairing requires no heuristics.** The trajectory layer's `state_before` and
  `state_after` references, together with the per-state graphs, supply the
  pairing directly. Only the labels had to be derived.

# Recommendations

Ordered by expected effect.

1. **Correct the attribution of scanned ports** in NSG-docker-state-creator
   (finding 19). Ports discovered by a scan should attach to the scanned host,
   not the scanning host. This is the single change that would most improve
   label quality, because it makes the effect of `FindServices` observable in
   the projected state.
2. **Establish whether causal isolation is achievable** (finding 16) by
   adjusting the settle window or the trajectory sensitivity. If it is not,
   effect-derived labels are permanently batch-level and command attribution
   must be the primary signal, which means every capture session should be
   driven from an interactive shell.
3. **Record sessions that change the projected state** (finding 20): a
   successful exploitation, data discovery on the newly controlled host, and a
   copy to a declared external host. These raise the identifiability limit,
   whereas further discovery-only sessions do not.
4. **Decide which extensions to NetSecGame are worthwhile** (finding 17). The
   strongest case is exfiltration parameterised by a path rather than by a
   discovered data object, followed by host discovery outside a known network,
   then control gained without prior service knowledge.
5. **Add source attribution to block nodes** (finding 8), which would make
   `known_blocks` projectable.
6. **Report the missing control check** in `generate_valid_actions` upstream
   (finding 18); it affects the simulator's agents as well as this repository.
