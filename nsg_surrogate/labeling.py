"""Turn an observed state transition into NetSecGame action labels.

Two signals, in the order they are trusted:

1. **Effect** — the NSG-level `StateDiff` between the projected states. Each of
   NSG's six knowledge categories can only be grown by one action type, so the
   diff yields both the type and (mostly) the parameters. This is primary
   because it is indifferent to *how* the operator did it: `nmap`, `masscan` or
   a hand-rolled loop all leave the same footprint.
2. **Command** — the `command.argv` on the action records attributed to the
   transition. Used to corroborate an effect label, to recover the two things a
   diff cannot see (which service was exploited, and who acted), and to label
   actions that changed nothing at all.

A label is only emitted if the action is in the candidate set of the *before*
state. An action the factored policy could never emit is not training data.

Everything that cannot be labelled is categorised rather than dropped: those
categories are the concrete argument for extending NetSecGame's action
vocabulary, so they are the module's second output alongside the labels.
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

#: Executable -> the NSG action type its use implies. Matched on the command's
#: basename; unlisted executables are off-vocabulary, which is a finding, not a
#: failure. Deliberately conservative: a wrong corroboration is worse than none.
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
    """One NSG action attributed to an observed transition."""

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
    """An observed change or command with no NetSecGame expression.

    `category` is the extension request: each distinct value names a capability
    NSG would need in order to represent what the operator actually did.
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
                "every NSG action needs a source_host, and none was projected",
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
                    f"{host} became known, but NSG can only learn hosts by scanning a "
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
                f"{network} became known; NSG has no action that discovers a network "
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
                    "itself became known; NSG requires the host to be known first",
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
                    f"{host} became controlled with no service known on it beforehand; NSG's "
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
                    f"data appeared on {host}, which the agent does not control; NSG's FindData "
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
                    f"{item.id} appeared on {destination}, which is not controlled; NSG's "
                    "ExfiltrateData requires a controlled destination",
                    dict(evidence_base, data=item.id, destination=str(destination)),
                )
            )
            continue
        if origin not in before.controlled_hosts:
            # Knowing where data sits is not being able to take it: the game
            # refuses exfiltration from a host the agent does not control, so
            # whatever really happened here had access NSG cannot represent.
            result.unmappable.append(
                Unmappable(
                    "data moved from an uncontrolled host",
                    f"{item.id} moved from {origin} to {destination}, but the agent does not "
                    f"control {origin}; NSG cannot exfiltrate from a host it has only learned "
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
                f"{host} learned it is blocked from {sorted(str(b) for b in blocked)}; NSG only "
                "produces known_blocks through the defender's BlockIP action",
                dict(evidence_base, host=str(host)),
            )
        )

    result.labels = merge_labels(result.labels)

    result.unmappable.extend(_blind_exfiltration(before, commands, source_host, evidence_base))

    # ── commands that changed nothing ─────────────────────────────────────────
    if diff.empty:
        result.labels.extend(
            _labels_from_commands_only(before, candidates, commands, source_host, evidence_base)
        )
        if not result.labels:
            for facts in commands:
                if not facts.action_types:
                    result.unmappable.append(
                        Unmappable(
                            "command outside the NSG action vocabulary",
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
) -> List[Label]:
    """Label actions that ran and yielded nothing.

    These matter: an NSG episode is full of actions that return no new
    knowledge, and a dataset built only from successful transitions would teach
    the surrogate that every action pays off.
    """
    labels: List[Label] = []
    for facts in commands:
        for action_type in sorted(facts.action_types, key=lambda t: t.name):
            action = _ground_command(facts, action_type, before, source_host)
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
                    notes=("no NSG-visible state change followed this command",),
                )
            )
    return merge_labels(labels)


def _blind_exfiltration(
    before: GameState,
    commands: Sequence[CommandFacts],
    source_host: IP,
    evidence_base: Dict[str, Any],
) -> List[Unmappable]:
    """Exfiltration of a file from a controlled host that was never discovered.

    An operator on a controlled host runs `scp /etc/shadow drop:/` without ever
    having enumerated the data. NSG builds its exfiltration action space over
    `known_data`, so there is no action for taking a file the agent has not
    found — even though controlling the host is exactly what should make it
    possible. These are recorded as extension requests, not invented candidates:
    an undiscovered file has no graph node for the policy's data head to score.
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
                    "file was never in known_data; NSG enumerates exfiltration over discovered "
                    "data only, so the attempt has no representation",
                    dict(evidence_base, path=path, command=facts.shell_text),
                )
            )
    return unmappable


def merge_labels(labels: Sequence[Label]) -> List[Label]:
    """Collapse labels that name the same action.

    Several commands in one batch ground to the same NSG action — `cat` recorded
    once by the shell and again by the exec fallback, or two probes of the same
    host. They are one label with several pieces of evidence, not several
    labels, and leaving them separate would weight that action by how noisily it
    was observed.
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
) -> Optional[Action]:
    """Build an action from a command's arguments, grounded in known state."""
    if action_type == ActionType.ScanNetwork:
        for cidr in facts.cidrs:
            for network in before.known_networks:
                if str(network) == cidr:
                    return Action(
                        ActionType.ScanNetwork,
                        {"source_host": source_host, "target_network": network},
                    )
        for address in facts.ips:
            network = _containing_network(IP(address), before.known_networks)
            if network is not None:
                return Action(
                    ActionType.ScanNetwork,
                    {"source_host": source_host, "target_network": network},
                )
        return None

    if action_type in (ActionType.FindServices, ActionType.ExploitService):
        for address in facts.ips:
            host = IP(address)
            if host not in before.known_hosts:
                continue
            if action_type == ActionType.FindServices:
                return Action(
                    ActionType.FindServices, {"source_host": source_host, "target_host": host}
                )
            services = sorted(before.known_services.get(host, ()), key=str)
            if services:
                return Action(
                    ActionType.ExploitService,
                    {
                        "source_host": source_host,
                        "target_host": host,
                        "target_service": services[0],
                    },
                )
        return None

    if action_type == ActionType.FindData:
        # The command ran on the host the trajectory was collected in, so the
        # acting host is the target. Picking any controlled host instead would
        # attribute a local `cat` to the declared exfiltration drop box.
        if source_host in before.controlled_hosts:
            return Action(
                ActionType.FindData, {"source_host": source_host, "target_host": source_host}
            )
        return None

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
                        return Action(
                            ActionType.ExfiltrateData,
                            {
                                "source_host": origin,
                                "target_host": destination,
                                "data": item,
                            },
                        )
        return None

    return None


# ── helpers ─────────────────────────────────────────────────────────────────


def infer_source_host(
    before: GameState, provenance: Optional[Provenance]
) -> Tuple[Optional[IP], Optional[str]]:
    """Which host acted.

    A trajectory is collected inside one container, so every observed action was
    issued from that container — the state creator's `host:local`. That is the
    assumption encoded here, and it is recorded as a note whenever the state
    offers more than one candidate, so a violation shows up in the data rather
    than silently mislabelling `source_host`.
    """
    if provenance is not None:
        local_controlled = sorted(
            provenance.local_hosts & set(before.controlled_hosts), key=str
        )
        if len(local_controlled) == 1:
            return local_controlled[0], None
        if len(local_controlled) > 1:
            return (
                local_controlled[0],
                f"{len(local_controlled)} local controlled hosts; assumed {local_controlled[0]}",
            )

    controlled = sorted(before.controlled_hosts, key=str)
    if not controlled:
        return None, None
    if len(controlled) == 1:
        return controlled[0], "source_host assumed from the single controlled host"
    return (
        controlled[0],
        f"no local host in provenance and {len(controlled)} controlled hosts; "
        f"assumed {controlled[0]}",
    )


def _pick_exploited_service(
    host: IP,
    known_services: Sequence[Service],
    commands: Sequence[CommandFacts],
    provenance: Optional[Provenance],
) -> Tuple[Service, Optional[str]]:
    """Recover which service was exploited; the state diff cannot say.

    Ports named on the command line are matched against the real ports the state
    creator observed (`Provenance.service_ports`). Failing that, the service
    name is matched against the executable (`ssh` -> the ssh service). Failing
    that, the canonical choice — the lexicographically smallest — keeps the label
    consistent with the policy's own exploit canonicalization, and the fallback
    is noted so these rows can be excluded from any service-level evaluation.
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
    """Why this action is not valid in the before-state, if it is not.

    The gate that keeps the dataset learnable: the factored policy can only ever
    emit actions from this set, so a label outside it is unusable supervision.
    """
    if contains(candidates, action):
        return None
    same_type = [candidate for candidate in candidates if candidate.type == action.type]
    if not same_type:
        return f"{action.type.name} is not valid in the before-state"
    return f"{action.type.name} parameters are not in the before-state candidate set"
