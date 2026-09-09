"""Assign a NetSecGame action to an observed state transition.

Two sources of evidence are used, in this order of precedence:

1. **Effect** — the `StateDiff` between the two projected states. Each of
   NetSecGame's six knowledge categories can be increased by exactly one action
   type, so the difference determines the action type and most of its
   parameters. This evidence is preferred because it does not depend on which
   tool the operator used: `nmap`, `masscan` and a shell loop produce the same
   change.
2. **Command** — the `command.argv` of the action records attributed to the
   transition. It confirms an effect-derived label, supplies the two parameters
   the difference cannot determine (which service was exploited, and which host
   acted), and labels actions that produced no observable change.

A label is emitted only if the action belongs to the candidate set of the
before-state. An action the policy could not select is not usable as a training
target.

Transitions that cannot be labelled are categorised rather than discarded. Each
category names a capability NetSecGame would need in order to represent the
observed behaviour, so the categories are the module's second output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import netaddr

from netsecgame.game_components import Action, ActionType, Data, GameState, IP, Network, Service

from .candidates import action_key, contains, enumerate_actions
from .state_adapter import Provenance
from .state_diff import StateDiff

#: Executable name to the action type its use implies, matched on the basename.
#: An executable that is not listed is recorded as outside the vocabulary, which
#: is a result rather than an error. The lists are deliberately narrow: a wrong
#: confirmation is worse than no confirmation.
COMMAND_RULES: Dict[ActionType, Tuple[str, ...]] = {
    ActionType.ScanNetwork: ("nmap", "masscan", "fping", "ping", "arp-scan", "netdiscover", "zmap"),
    ActionType.FindServices: ("nmap", "nc", "ncat", "netcat", "telnet", "nikto", "whatweb", "curl", "wget"),
    ActionType.ExploitService: (
        "ssh", "sshpass", "hydra", "medusa", "sqlmap", "msfconsole", "msfvenom",
        "psexec.py", "smbclient", "mysql", "psql", "redis-cli", "exploit",
    ),
    ActionType.FindData: ("find", "ls", "cat", "grep", "strings", "locate", "head", "tail", "less"),
    ActionType.ExfiltrateData: ("scp", "rsync", "sftp", "nc", "ncat", "curl", "wget", "base64", "tar"),
}

#: Scanning a range and probing one host both use nmap; the flags disambiguate.
_HOST_SCAN_FLAGS = ("-sn", "-sP", "-PE")
_SERVICE_SCAN_FLAGS = ("-p", "-sV", "-sS", "-sT", "-A")
_UPLOAD_FLAGS = ("-T", "--upload-file", "--data-binary", "-d")

_CIDR = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})/(\d{1,2})\b")
_IPV4 = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")


@dataclass(frozen=True)
class Label:
    """One NetSecGame action attributed to an observed transition."""

    action: Action
    label_source: str  # effect | effect+command | command
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.as_dict,
            "action_type": self.action.type.name,
            "label_source": self.label_source,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Unmappable:
    """An observed change or command with no NetSecGame representation.

    Each distinct `category` names a capability NetSecGame would need in order to
    express what was observed.
    """

    category: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"category": self.category, "detail": self.detail, "evidence": self.evidence}


@dataclass
class LabelResult:
    labels: List[Label] = field(default_factory=list)
    unmappable: List[Unmappable] = field(default_factory=list)
    #: Labels rejected because the action is not in the before-state candidates.
    rejected: List[Tuple[Label, str]] = field(default_factory=list)


# ── command parsing ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CommandFacts:
    """What a command line asserts, before any state grounding."""

    argv: Tuple[str, ...]
    executable: str
    action_types: FrozenSet[ActionType]
    ips: Tuple[str, ...]
    cidrs: Tuple[str, ...]
    paths: Tuple[str, ...]
    ports: Tuple[int, ...]

    @property
    def shell_text(self) -> str:
        return " ".join(self.argv)


def parse_command(record: Dict[str, Any]) -> Optional[CommandFacts]:
    """Extract the assertable facts from one action record's command."""
    command = record.get("command") or {}
    argv = tuple(str(part) for part in (command.get("argv") or ()))
    if not argv:
        return None

    executable = str(command.get("executable") or argv[0]).rsplit("/", 1)[-1]
    joined = " ".join(argv)

    types = {
        action_type
        for action_type, executables in COMMAND_RULES.items()
        if executable in executables
    }
    # nmap covers both discovery and service enumeration; the flags decide.
    if executable in ("nmap", "masscan"):
        if any(flag in argv for flag in _HOST_SCAN_FLAGS):
            types.discard(ActionType.FindServices)
        elif any(argument.startswith(_SERVICE_SCAN_FLAGS) for argument in argv):
            types.discard(ActionType.ScanNetwork)
    # curl/wget/nc read by default and write only when told to.
    if executable in ("curl", "wget", "nc", "ncat"):
        if not any(flag in argv for flag in _UPLOAD_FLAGS):
            types.discard(ActionType.ExfiltrateData)

    cidrs = tuple(f"{net}/{mask}" for net, mask in _CIDR.findall(joined))
    ips = tuple(
        address
        for address in _IPV4.findall(joined)
        if not any(address in cidr for cidr in cidrs)
    )
    paths = tuple(
        argument
        for argument in argv[1:]
        if not argument.startswith("-")
        and argument not in cidrs
        and (argument.startswith(("/", "./", "../")) or "/" in argument)
    )

    return CommandFacts(
        argv=argv,
        executable=executable,
        action_types=frozenset(types),
        ips=ips,
        cidrs=cidrs,
        paths=paths,
        ports=_extract_ports(argv),
    )


def _extract_ports(argv: Sequence[str]) -> Tuple[int, ...]:
    """Ports named on the command line: `-p22`, `-p 22,443`, `--port=8080`."""
    ports: List[int] = []
    for index, argument in enumerate(argv):
        raw: Optional[str] = None
        if argument.startswith("--port="):
            raw = argument.split("=", 1)[1]
        elif argument in ("-p", "-P", "--port") and index + 1 < len(argv):
            raw = argv[index + 1]
        elif re.fullmatch(r"-p[\d,]+", argument):
            raw = argument[2:]
        if not raw:
            continue
        for part in raw.split(","):
            if part.isdigit() and 1 <= int(part) <= 65535:
                ports.append(int(part))
    return tuple(dict.fromkeys(ports))


# ── effect-based labelling ──────────────────────────────────────────────────


def label_transition(
    before: GameState,
    after: GameState,
    diff: StateDiff,
    commands: Sequence[CommandFacts] = (),
    provenance: Optional[Provenance] = None,
    docker_evidence: Optional[Dict[str, Any]] = None,
    states_differ: Optional[bool] = None,
) -> LabelResult:
    """Label one observed transition, and categorise what cannot be labelled."""
    result = LabelResult()
    candidates = enumerate_actions(before, canonicalize_exploit_services=False)
    source_host, source_note = infer_source_host(before, provenance)
    evidence_base = dict(docker_evidence or {})
    if commands:
        evidence_base["commands"] = [facts.shell_text for facts in commands[:8]]

    if source_host is None:
        result.unmappable.append(
            Unmappable(
                "no controlled source host",
                "every NetSecGame action needs a source_host, and none was projected",
                evidence_base,
            )
        )
        return result

    def emit(
        action: Action,
        implied_type: ActionType,
        confidence: float,
        notes: Iterable[str] = (),
        extra_evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        corroborating = [facts for facts in commands if implied_type in facts.action_types]
        evidence = dict(evidence_base)
        if extra_evidence:
            evidence.update(extra_evidence)
        if corroborating:
            evidence["corroborating_commands"] = [facts.shell_text for facts in corroborating[:4]]
            label_source, weight = "effect+command", 1.0
        else:
            label_source, weight = "effect", 0.6
        all_notes = list(notes)
        if source_note:
            all_notes.append(source_note)
        if not corroborating and commands:
            all_notes.append("no command in this batch matches the implied action type")

        label = Label(
            action=action,
            label_source=label_source,
            confidence=round(confidence * weight, 3),
            evidence=evidence,
            notes=tuple(all_notes),
        )
        reason = _rejection_reason(action, candidates)
        if reason:
            result.rejected.append((label, reason))
        else:
            result.labels.append(label)

    # ── ScanNetwork: new hosts or networks appeared ───────────────────────────
    for host in sorted(diff.new_hosts, key=str):
        network = _containing_network(host, before.known_networks)
        if network is None:
            result.unmappable.append(
                Unmappable(
                    "host discovered outside any known network",
                    f"{host} became known, but NetSecGame can only learn hosts by scanning a "
                    "known network — there is no action for learning a host from a DNS "
                    "lookup, a log file, or an outbound connection",
                    dict(evidence_base, host=str(host)),
                )
            )
            continue
        emit(
            Action(ActionType.ScanNetwork, {"source_host": source_host, "target_network": network}),
            ActionType.ScanNetwork,
            0.9,
            extra_evidence={"new_host": str(host), "network": str(network)},
        )

    for network in sorted(diff.new_networks, key=str):
        result.unmappable.append(
            Unmappable(
                "network learned without a scan",
                f"{network} became known; NetSecGame has no action that discovers a network "
                "(they come from the scenario's starting knowledge)",
                dict(evidence_base, network=str(network)),
            )
        )

    # ── FindServices: services appeared on a host ─────────────────────────────
    for host, services in sorted(diff.new_services.items(), key=lambda kv: str(kv[0])):
        if host not in before.known_hosts:
            result.unmappable.append(
                Unmappable(
                    "services learned on a previously unknown host",
                    f"services appeared on {host} in the same transition in which the host "
                    "itself became known; NetSecGame requires the host to be known first",
                    dict(evidence_base, host=str(host), services=[str(s) for s in services]),
                )
            )
            continue
        emit(
            Action(ActionType.FindServices, {"source_host": source_host, "target_host": host}),
            ActionType.FindServices,
            0.9,
            extra_evidence={"host": str(host), "new_services": sorted(str(s) for s in services)},
        )

    # ── ExploitService: a host became controlled ──────────────────────────────
    for host in sorted(diff.new_controlled, key=str):
        known_services = sorted(before.known_services.get(host, ()), key=str)
        if not known_services:
            result.unmappable.append(
                Unmappable(
                    "control gained without prior service knowledge",
                    f"{host} became controlled with no service known on it beforehand; NetSecGame's "
                    "ExploitService requires a known service, so credential reuse, key-based "
                    "login and lateral movement have no representation",
                    dict(evidence_base, host=str(host)),
                )
            )
            continue

        service, service_note = _pick_exploited_service(
            host, known_services, commands, provenance
        )
        emit(
            Action(
                ActionType.ExploitService,
                {"source_host": source_host, "target_host": host, "target_service": service},
            ),
            ActionType.ExploitService,
            0.85,
            notes=(service_note,) if service_note else (),
            extra_evidence={"host": str(host), "service": str(service)},
        )

    # ── FindData: data appeared on a host ─────────────────────────────────────
    for host, items in sorted(diff.new_data.items(), key=lambda kv: str(kv[0])):
        if host not in before.controlled_hosts:
            result.unmappable.append(
                Unmappable(
                    "data learned on an uncontrolled host",
                    f"data appeared on {host}, which the agent does not control; NetSecGame's FindData "
                    "only enumerates data on controlled hosts, so reading a remote page, an "
                    "open share or a banner has no representation",
                    dict(evidence_base, host=str(host), data=sorted(item.id for item in items)),
                )
            )
            continue
        emit(
            Action(ActionType.FindData, {"source_host": host, "target_host": host}),
            ActionType.FindData,
            0.9,
            extra_evidence={"host": str(host), "new_data": sorted(item.id for item in items)},
        )

    # ── ExfiltrateData: a copy of known data appeared elsewhere ───────────────
    for item, origin, destination in diff.relocated_data:
        if destination not in after.controlled_hosts:
            result.unmappable.append(
                Unmappable(
                    "data copied to an uncontrolled host",
                    f"{item.id} appeared on {destination}, which is not controlled; NetSecGame's "
                    "ExfiltrateData requires a controlled destination",
                    dict(evidence_base, data=item.id, destination=str(destination)),
                )
            )
            continue
        if origin not in before.controlled_hosts:
            # Knowing where data sits is not being able to take it: the game
            # refuses exfiltration from a host the agent does not control, so
            # whatever really happened here had access NetSecGame cannot represent.
            result.unmappable.append(
                Unmappable(
                    "data moved from an uncontrolled host",
                    f"{item.id} moved from {origin} to {destination}, but the agent does not "
                    f"control {origin}; NetSecGame cannot exfiltrate from a host it has only learned "
                    "about, so the access that made this possible is unrepresented",
                    dict(evidence_base, data=item.id, origin=str(origin)),
                )
            )
            continue
        emit(
            Action(
                ActionType.ExfiltrateData,
                {"source_host": origin, "target_host": destination, "data": item},
            ),
            ActionType.ExfiltrateData,
            0.85,
            extra_evidence={"data": item.id, "from": str(origin), "to": str(destination)},
        )

    # ── blocks: defender vocabulary ───────────────────────────────────────────
    for host, blocked in sorted(diff.new_blocks.items(), key=lambda kv: str(kv[0])):
        result.unmappable.append(
            Unmappable(
                "firewall knowledge gained",
                f"{host} learned it is blocked from {sorted(str(b) for b in blocked)}; NetSecGame only "
                "produces known_blocks through the defender's BlockIP action",
                dict(evidence_base, host=str(host)),
            )
        )

    result.labels = merge_labels(result.labels)

    result.unmappable.extend(_blind_exfiltration(before, commands, source_host, evidence_base))
    result.unmappable.extend(_remote_execution(before, commands, evidence_base))
    result.unmappable.extend(_unmappable_scan_ranges(before, commands, evidence_base))

    # ── commands that changed nothing ─────────────────────────────────────────
    if diff.empty:
        result.labels.extend(
            _labels_from_commands_only(
                before,
                candidates,
                commands,
                source_host,
                evidence_base,
                states_differ=bool(states_differ),
            )
        )
        if not result.labels:
            for facts in commands:
                if not facts.action_types:
                    result.unmappable.append(
                        Unmappable(
                            "command outside the NetSecGame action vocabulary",
                            f"{facts.executable}: {facts.shell_text[:120]}",
                            dict(evidence_base, executable=facts.executable),
                        )
                    )

    result.labels = merge_labels(result.labels)
    result.labels.sort(key=lambda label: (-label.confidence, label.action.type.name))
    return result


def _labels_from_commands_only(
    before: GameState,
    candidates: Sequence[Action],
    commands: Sequence[CommandFacts],
    source_host: IP,
    evidence_base: Dict[str, Any],
    states_differ: bool = False,
) -> List[Label]:
    """Label actions that ran without producing an observable change.

    These are necessary to the dataset. A NetSecGame episode contains many
    actions that return no new knowledge, and a dataset built only from
    state-changing transitions would represent every action as successful.

    `states_differ` separates two genuinely different situations. When the
    observation never advanced, the action really did nothing. When the docker
    states differ but their NetSecGame projections are identical, the action *did*
    something the mapping cannot see — `nmap -sS` against a host produces 1000
    service nodes that the state creator attributes to the scanning host and the
    adapter then filters, so a successful service scan leaves no NetSecGame trace. The
    note says which case a label came from, because the second is a mapping
    problem rather than a property of the action.
    """
    labels: List[Label] = []
    for facts in commands:
        for action_type in sorted(facts.action_types, key=lambda t: t.name):
            action, grounding_notes = _ground_command(
                facts, action_type, before, source_host
            )
            if action is None:
                continue
            if _rejection_reason(action, candidates):
                continue
            labels.append(
                Label(
                    action=action,
                    label_source="command",
                    confidence=0.5,
                    evidence=dict(evidence_base, command=facts.shell_text),
                    notes=(
                        (
                            "docker state advanced but its NetSecGame projection is unchanged: "
                            "the effect of this command is invisible to the mapping"
                            if states_differ
                            else "no state change followed this command"
                        ),
                    )
                    + grounding_notes,
                )
            )
    return merge_labels(labels)


def _blind_exfiltration(
    before: GameState,
    commands: Sequence[CommandFacts],
    source_host: IP,
    evidence_base: Dict[str, Any],
) -> List[Unmappable]:
    """Copying a file from a controlled host without having discovered it.

    An operator on a controlled host runs `scp /etc/shadow drop:/` without having
    enumerated the data first. NetSecGame builds its exfiltration action space
    over `known_data`, so there is no action for copying a file the agent has not
    discovered, even though controlling the host is what makes it possible.
    These are recorded rather than turned into candidates: an undiscovered file
    has no graph node for the policy's data head to score.
    """
    if source_host not in before.controlled_hosts:
        return []

    known_ids = {item.id for items in before.known_data.values() for item in items}
    unmappable: List[Unmappable] = []
    for facts in commands:
        if ActionType.ExfiltrateData not in facts.action_types:
            continue
        for path in facts.paths:
            if path in known_ids or any(path in known or known in path for known in known_ids):
                continue
            unmappable.append(
                Unmappable(
                    "blind exfiltration of undiscovered data",
                    f"{facts.executable} moved {path} from controlled {source_host}, but the "
                    "file was never in known_data; NetSecGame enumerates exfiltration over discovered "
                    "data only, so the attempt has no representation",
                    dict(evidence_base, path=path, command=facts.shell_text),
                )
            )
    return unmappable


def merge_labels(labels: Sequence[Label]) -> List[Label]:
    """Combine labels that name the same action.

    Several commands attributed to one transition can produce the same action:
    `cat` recorded once by the shell collector and again by the exec fallback, or
    two probes of the same host. These are one label with several pieces of
    evidence. Keeping them separate would weight the action by how often it was
    observed rather than by whether it occurred.
    """
    merged: Dict[Any, Label] = {}
    order: List[Any] = []

    for label in labels:
        key = action_key(label.action)
        incumbent = merged.get(key)
        if incumbent is None:
            merged[key] = label
            order.append(key)
            continue

        commands = list(incumbent.evidence.get("grounded_commands", []))
        for source in (incumbent, label):
            command = source.evidence.get("command")
            if command and command not in commands:
                commands.append(command)
        best = incumbent if incumbent.confidence >= label.confidence else label
        evidence = dict(best.evidence)
        if commands:
            evidence["grounded_commands"] = commands
            evidence.pop("command", None)
        merged[key] = Label(
            action=best.action,
            label_source=best.label_source,
            confidence=max(incumbent.confidence, label.confidence),
            evidence=evidence,
            notes=tuple(dict.fromkeys(incumbent.notes + label.notes)),
        )

    return [merged[key] for key in order]


def _ground_command(
    facts: CommandFacts,
    action_type: ActionType,
    before: GameState,
    source_host: IP,
) -> Tuple[Optional[Action], Tuple[str, ...]]:
    """Build an action from a command's arguments, grounded in known state.

    Returns the action and any notes about granularity the mapping had to
    flatten — an operator's command rarely lines up exactly with an NetSecGame action.
    """
    if action_type == ActionType.ScanNetwork:
        for cidr in facts.cidrs:
            network, note = _network_for_scanned_range(cidr, before.known_networks)
            if network is not None:
                return (
                    Action(
                        ActionType.ScanNetwork,
                        {"source_host": source_host, "target_network": network},
                    ),
                    (note,) if note else (),
                )
        for address in facts.ips:
            network = _containing_network(IP(address), before.known_networks)
            if network is not None:
                return (
                    Action(
                        ActionType.ScanNetwork,
                        {"source_host": source_host, "target_network": network},
                    ),
                    (f"operator probed the single host {address}; NetSecGame can only scan a network",),
                )
        return None, ()

    if action_type in (ActionType.FindServices, ActionType.ExploitService):
        for address in facts.ips:
            host = IP(address)
            if host not in before.known_hosts:
                continue
            if action_type == ActionType.FindServices:
                return (
                    Action(
                        ActionType.FindServices,
                        {"source_host": source_host, "target_host": host},
                    ),
                    (),
                )
            services = sorted(before.known_services.get(host, ()), key=str)
            if services:
                matched = _service_for_ports(host, services, facts, before)
                return (
                    Action(
                        ActionType.ExploitService,
                        {
                            "source_host": source_host,
                            "target_host": host,
                            "target_service": matched or services[0],
                        },
                    ),
                    () if matched else ("exploited service not identifiable from the command",),
                )
        return None, ()

    if action_type == ActionType.FindData:
        # The command ran on the host the trajectory was collected in, so the
        # acting host is the target. Picking any controlled host instead would
        # attribute a local `cat` to the declared exfiltration drop box.
        if source_host in before.controlled_hosts:
            return (
                Action(
                    ActionType.FindData,
                    {"source_host": source_host, "target_host": source_host},
                ),
                (),
            )
        return None, ()

    if action_type == ActionType.ExfiltrateData:
        for address in facts.ips:
            destination = IP(address)
            if destination not in before.controlled_hosts:
                continue
            for origin, items in sorted(before.known_data.items(), key=lambda kv: str(kv[0])):
                if origin == destination:
                    continue
                for item in sorted(items, key=str):
                    if not facts.paths or any(item.id in path or path in item.id for path in facts.paths):
                        return (
                            Action(
                                ActionType.ExfiltrateData,
                                {
                                    "source_host": origin,
                                    "target_host": destination,
                                    "data": item,
                                },
                            ),
                            (),
                        )
        return None, ()

    return None, ()


# ── helpers ─────────────────────────────────────────────────────────────────


def infer_source_host(
    before: GameState, provenance: Optional[Provenance]
) -> Tuple[Optional[IP], Optional[str]]:
    """Determine which host issued the action.

    A recording is produced by an observer inside one container, so the commands
    it captures were executed in that container, which the state creator
    identifies as `host:local`. That host is therefore preferred.

    It does not follow that a trajectory concerns a single machine. An agent that
    gains control of a second host acts from there as well, and the two ways that
    happens are attributed differently:

    * a second observer on that host produces its own recording, in which its
      own container is `host:local`, so the attribution is correct per recording;
    * the agent issues commands through a remote-execution wrapper such as
      `ssh host "..."`. The observer sees the wrapper but not what it ran, so the
      inner action's host cannot be recovered from this evidence.

    Consequently the choice is recorded as a note whenever the state contains
    more than one controlled host, even when a local host is identified: the
    recording cannot establish that the local host was the one that acted. Where
    the host cannot be determined at all, `None` is returned and the transition
    is reported as unmappable rather than attributed arbitrarily.
    """
    controlled = sorted(before.controlled_hosts, key=str)
    if not controlled:
        return None, None

    local_controlled = (
        sorted(provenance.local_hosts & set(before.controlled_hosts), key=str)
        if provenance is not None
        else []
    )

    if len(local_controlled) == 1:
        chosen = local_controlled[0]
        if len(controlled) == 1:
            return chosen, None
        return (
            chosen,
            f"{len(controlled)} controlled hosts; attributed to the observed host "
            f"{chosen}, which the recording cannot confirm acted",
        )

    if len(local_controlled) > 1:
        return (
            local_controlled[0],
            f"{len(local_controlled)} controlled hosts are marked local; "
            f"attributed to {local_controlled[0]}",
        )

    if len(controlled) == 1:
        return controlled[0], "attributed to the only controlled host"
    return (
        controlled[0],
        f"no observed host among {len(controlled)} controlled hosts; "
        f"attributed to {controlled[0]}",
    )


#: Executables that run a command on another host. The observer records the
#: wrapper, not what it executed there.
REMOTE_EXECUTION = frozenset(
    {"ssh", "sshpass", "psexec.py", "smbclient", "winrs", "kubectl", "docker"}
)


def _remote_execution(
    before: GameState, commands: Sequence[CommandFacts], evidence_base: Dict[str, Any]
) -> List[Unmappable]:
    """Commands that ran on a host this recording does not observe.

    `ssh controlled-host "nmap ..."` performs an action whose `source_host` is
    the remote host, but the observer sees only the wrapper: the inner command,
    and any state it changed there, are outside this recording. Attributing such
    an action to the observed host would name the wrong source, so the case is
    recorded instead.

    A wrapper naming a host that is *not* yet controlled is an access attempt
    rather than remote execution, and is labelled as `ExploitService` by the
    ordinary rules.
    """
    unmappable: List[Unmappable] = []
    for facts in commands:
        if facts.executable not in REMOTE_EXECUTION:
            continue
        for address in facts.ips:
            host = IP(address)
            if host not in before.controlled_hosts:
                continue
            unmappable.append(
                Unmappable(
                    "command executed on another host",
                    f"{facts.executable} ran a command on {host}, which the agent already "
                    "controls; the action taken there has a different source_host and is "
                    "not observed by this recording",
                    dict(evidence_base, host=str(host), command=facts.shell_text),
                )
            )
    return unmappable


def _pick_exploited_service(
    host: IP,
    known_services: Sequence[Service],
    commands: Sequence[CommandFacts],
    provenance: Optional[Provenance],
) -> Tuple[Service, Optional[str]]:
    """Determine which service was exploited, which the state difference cannot.

    A port named on the command line is matched against the ports the state
    creator observed (`Provenance.service_ports`). Otherwise the service name is
    matched against the executable, so `ssh` selects the ssh service. Otherwise
    the lexicographically smallest service is used, which matches the policy's
    own reduction of `ExploitService`; that case is recorded as a note so the
    affected rows can be excluded from any evaluation of the service parameter.
    """
    if provenance is not None:
        wanted_ports = {port for facts in commands for port in facts.ports}
        if wanted_ports:
            for service in known_services:
                port = provenance.service_ports.get((host, service))
                if port in wanted_ports:
                    return service, None

    executables = {facts.executable for facts in commands}
    for service in known_services:
        if service.name.lower() in executables:
            return service, None

    return known_services[0], "exploited service not recoverable; canonical service used"


def _network_for_scanned_range(
    cidr: str, networks: Iterable[Network]
) -> Tuple[Optional[Network], Optional[str]]:
    """Map a scanned address range onto a known NetSecGame network.

    Operators do not necessarily scan the networks NetSecGame knows about. In
    the strategic run the container knows `172.23.0.0/16` and the operator
    scanned `172.23.0.0/24`, a sub-range. `ScanNetwork` takes a known `Network`
    object, so the containing network is the only expressible target, and the
    difference in granularity is recorded as a note. A range that no known
    network contains is not mapped.
    """
    try:
        scanned = ipaddress.IPv4Network(cidr, strict=False)
    except ValueError:
        return None, None

    parsed: List[Tuple[ipaddress.IPv4Network, Network]] = []
    for network in networks:
        try:
            parsed.append((ipaddress.IPv4Network(str(network), strict=False), network))
        except ValueError:
            continue

    for candidate, network in parsed:
        if candidate == scanned:
            return network, None

    containing = [(c, n) for c, n in parsed if scanned.subnet_of(c)]
    if containing:
        # Narrowest containing network is the closest expressible target.
        candidate, network = max(containing, key=lambda pair: pair[0].prefixlen)
        return (
            network,
            f"operator scanned {scanned}, a subrange of known network {network}; "
            "NetSecGame can only scan a whole known network",
        )
    return None, None


def _service_for_ports(
    host: IP,
    services: Sequence[Service],
    facts: CommandFacts,
    before: GameState,
) -> Optional[Service]:
    """Match a service by a port named on the command line, else by name."""
    if facts.ports:
        for service in services:
            for token in (service.name, f"{service.type}/{service.name}"):
                digits = "".join(character for character in str(token) if character.isdigit())
                if digits and int(digits) in facts.ports:
                    return service
    for service in services:
        if service.name.lower() == facts.executable.lower():
            return service
    return None


def _unmappable_scan_ranges(
    before: GameState, commands: Sequence[CommandFacts], evidence_base: Dict[str, Any]
) -> List[Unmappable]:
    """Scans of ranges no known network contains."""
    unmappable: List[Unmappable] = []
    for facts in commands:
        if ActionType.ScanNetwork not in facts.action_types:
            continue
        for cidr in facts.cidrs:
            network, _ = _network_for_scanned_range(cidr, before.known_networks)
            if network is None:
                unmappable.append(
                    Unmappable(
                        "scan of a range that is not a known network",
                        f"{facts.executable} scanned {cidr}, which no known network contains; "
                        "NetSecGame's ScanNetwork is parameterized by a known network object",
                        dict(evidence_base, cidr=cidr, command=facts.shell_text),
                    )
                )
    return unmappable


def _containing_network(host: IP, networks: Iterable[Network]) -> Optional[Network]:
    """The known network a host falls inside, narrowest first."""
    matches = []
    for network in networks:
        try:
            if str(host) in netaddr.IPNetwork(str(network)):
                matches.append(network)
        except netaddr.AddrFormatError:
            continue
    if not matches:
        return None
    return sorted(matches, key=lambda net: (-int(net.mask), str(net)))[0]


def _rejection_reason(action: Action, candidates: Sequence[Action]) -> Optional[str]:
    """Why this action is not valid in the before-state, or `None` if it is.

    The policy can only select actions from the candidate set, so a label
    outside that set cannot be used as a training target.
    """
    if contains(candidates, action):
        return None
    same_type = [candidate for candidate in candidates if candidate.type == action.type]
    if not same_type:
        return f"{action.type.name} is not valid in the before-state"
    return f"{action.type.name} parameters are not in the before-state candidate set"
