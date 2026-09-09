"""Project an NSG-state-creator state graph into a netsecgame `GameState`.

Input is `state/graph.json` as written by NSG-docker-state-creator at any detail
level (`strategic` is the intended one; schema `nsg-state-graph/1.1`). Output is
a `GameState` plus a report of everything the projection had to drop or decide,
because in this direction the source graph is strictly richer than the target:

    docker graph                     netsecgame GameState
    ------------------------------   -----------------------------------------
    host / network / service / data  known_hosts, known_networks,
                                     known_services, known_data
    CONTROLS edge                    controlled_hosts
    block                            (no faithful target — see report notes)
    program / command / user         (no target at all)
    confidence, evidence, inferred   (no target — collapsed to boolean fact)

The projection adds no facts. An entity that cannot be represented is discarded
and recorded with the reason, and any choice the observation does not determine
is reported as such. The full graph is read rather than `summary.json`, because
the summary omits service ports, per-node confidence and host locality, which
the encoder uses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import ipaddress
import json
import os
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from netsecgame.game_components import Data, GameState, IP, Network, Service

SUPPORTED_GRAPH_SCHEMAS = ("nsg-state-graph/1.1",)

DEFAULT_AGENT_ID = "agent:observed"

# Edge types that record what the agent knows. A NetSecGame state is defined as
# the agent's knowledge, so these edges select the subgraph to project.
KNOWLEDGE_EDGES = {
    "KNOWS_HOST": "host",
    "KNOWS_NETWORK": "network",
    "KNOWS_SERVICE": "service",
    "KNOWS_DATA": "data",
    "KNOWS_BLOCK": "block",
}
CONTROL_EDGE = "CONTROLS"

# Node types with no representation anywhere in GameState.
UNREPRESENTABLE_NODE_TYPES = frozenset({"program", "command", "user", "agent", "observation"})

# Paths that hold operating-system state rather than data an agent would target.
# The state creator applies its own filter at the strategic level; this list is
# retained because the operational and forensic levels do not.
DEFAULT_DATA_PATH_DENYLIST: Tuple[str, ...] = (
    r"^/tmp/",
    r"^/var/tmp/",
    r"^/var/lib/(apt|dpkg)/",
    r"^/var/cache/",
    r"^/var/log/",
    r"^/proc/",
    r"^/sys/",
    r"^/dev/",
    r"^/run/",
    r"^/observation/",
)


@dataclass(frozen=True)
class AdapterConfig:
    """Options that decide how much of the state graph is projected.

    Every default encodes a judgement that the observation does not determine,
    so each is an option rather than a constant. Lowering `min_confidence` or
    clearing `data_path_denylist` shows how much of the projected action space
    depends on those judgements.
    """

    #: Drop nodes the state creator is less sure about than this. Inferred
    #: networks arrive at 0.35 and block hypotheses at 0.78.
    min_confidence: float = 0.5
    #: Keep `/24`-estimated networks (confidence 0.35, `inferred=true`).
    include_inferred_networks: bool = False
    #: Networks that are routes, not attack surface.
    excluded_cidrs: FrozenSet[str] = frozenset({"0.0.0.0/0", "::/0"})
    #: Statuses indicating that a service node records the agent's own traffic
    #: rather than a service. A 1000-port `nmap -sS` produces 999 nodes with
    #: `status: "attempted"` (confidence 0.7, `zeek_state: S0`), attributed to
    #: the scanning host; projecting them would populate `known_services` with
    #: the agent's own scan. A node that also carries positive evidence
    #: (`service_status_positive`) is kept, as is a node with no status at all,
    #: since a missing optional field is not evidence of absence. An empty set
    #: disables the filter.
    service_status_denylist: FrozenSet[str] = frozenset({"attempted", "targeted"})
    #: Statuses that are direct evidence a service exists; these override the
    #: denylist when a node carries both (`["listening", "attempted"]`).
    service_status_positive: FrozenSet[str] = frozenset({"open", "listening", "observed"})
    #: Drop data whose existence the observer could not confirm (created and
    #: then deleted files arrive as `existence: "unknown"`).
    require_confirmed_data: bool = True
    #: Treat a file named on the operator's command line as discovered data even
    #: when the filesystem monitor never reconciled it. See `_data_is_present`.
    accept_command_argument_data: bool = True
    data_path_denylist: Tuple[str, ...] = DEFAULT_DATA_PATH_DENYLIST
    #: Cap per host; NetSecGame's ExfiltrateData action space is |data| x |controlled|.
    max_data_per_host: int = 32
    #: Networks whose hosts are targets. The state creator records every host
    #: the container contacted, which in the sample run includes Ubuntu archive
    #: mirrors, Cloudflare and public resolvers. NetSecGame treats every known
    #: host as attackable, so leaving this empty presents third-party internet
    #: hosts to the policy as valid targets.
    scope_cidrs: FrozenSet[str] = frozenset()
    #: Hosts outside the range, and therefore valid exfiltration destinations.
    #: Replaces the simulator's `is_private()` test, which carries no
    #: information in an all-RFC1918 container network. Exempt from
    #: `scope_cidrs`.
    external_hosts: FrozenSet[str] = frozenset()
    #: Add declared external hosts to `controlled_hosts`. The observed container
    #: is the only host the state creator can report as controlled, and
    #: `ExfiltrateData` requires a second controlled host as the destination, so
    #: without this the action is unreachable and no scenario goal can be met.
    #: The addition is configuration rather than observation, and is always
    #: recorded as such in the report.
    treat_external_as_controlled: bool = True
    #: Agent node whose knowledge edges define the projectable subgraph.
    agent_id: str = DEFAULT_AGENT_ID


@dataclass(frozen=True)
class Drop:
    """One entity the projection did not represent, and why."""

    category: str
    node_id: str
    reason: str
    label: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        shown = self.label or self.node_id
        return f"{self.category}: {shown} ({self.reason})"


@dataclass
class AdapterReport:
    """What the projection read, what it produced, and what it discarded."""

    schema_version: str = ""
    detail_level: str = ""
    generated_at: Optional[float] = None
    source: str = ""
    nodes_in: Dict[str, int] = field(default_factory=dict)
    edges_in: Dict[str, int] = field(default_factory=dict)
    counts_out: Dict[str, int] = field(default_factory=dict)
    drops: List[Drop] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def dropped_by_category(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for drop in self.drops:
            out[drop.category] += 1
        return dict(out)

    @property
    def dropped_by_reason(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for drop in self.drops:
            out[f"{drop.category}/{drop.reason}"] += 1
        return dict(out)


@dataclass
class Provenance:
    """Information the encoder needs that `GameState` cannot hold.

    Discarding it would either remove signal from the policy (service ports,
    external hosts) or make a selected action impossible to trace back to the
    observation it came from (`node_id`).
    """

    #: NetSecGame object -> originating docker node id.
    node_id: Dict[Any, str] = field(default_factory=dict)
    #: (host, service) -> TCP/UDP port, which NetSecGame's `Service` has no field for.
    service_ports: Dict[Tuple[IP, Service], int] = field(default_factory=dict)
    #: Hosts the state creator saw as the local container.
    local_hosts: Set[IP] = field(default_factory=set)
    #: Hosts designated as outside the range (exfiltration destinations).
    external_hosts: Set[IP] = field(default_factory=set)
    #: Per-node confidence, keyed the same way as `node_id`.
    confidence: Dict[Any, float] = field(default_factory=dict)
    #: Every address the state creator attributed to a projected host. NetSecGame
    #: allows exactly one IP per host, so multi-homed containers lose the rest.
    host_addresses: Dict[IP, Tuple[str, ...]] = field(default_factory=dict)
    #: Block nodes verbatim; see `AdapterReport.notes` for why they are unmapped.
    blocks: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Projection:
    state: GameState
    report: AdapterReport
    provenance: Provenance


def load_graph(path: str) -> Tuple[Dict[str, Any], str]:
    """Load a state graph from a file, a state dir, or an observation root."""
    candidates: Sequence[str]
    if os.path.isdir(path):
        candidates = (
            os.path.join(path, "graph.json"),
            os.path.join(path, "state", "graph.json"),
        )
    else:
        candidates = (path,)

    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as handle:
                return json.load(handle), candidate

    raise FileNotFoundError(f"no state graph found at {path} (tried: {', '.join(candidates)})")


def project_graph_to_game_state(
    graph: Dict[str, Any],
    config: Optional[AdapterConfig] = None,
    source: str = "",
) -> Projection:
    """Project a state graph into a `GameState`, reporting every loss."""
    config = config or AdapterConfig()
    report = AdapterReport(
        schema_version=str(graph.get("schema_version", "")),
        detail_level=str(graph.get("detail_level", "")),
        generated_at=graph.get("generated_at"),
        source=source,
    )
    provenance = Provenance()

    if report.schema_version and report.schema_version not in SUPPORTED_GRAPH_SCHEMAS:
        report.notes.append(
            f"unverified graph schema {report.schema_version!r}; "
            f"adapter was written against {', '.join(SUPPORTED_GRAPH_SCHEMAS)}"
        )
    if report.detail_level and report.detail_level != "strategic":
        report.notes.append(
            f"graph detail level is {report.detail_level!r}; the surrogate targets "
            "'strategic', which drops process/flow detail and keeps only important files"
        )

    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        node_id = node.get("id")
        if node_id:
            nodes_by_id[node_id] = node
        report.nodes_in[str(node.get("type", "?"))] = (
            report.nodes_in.get(str(node.get("type", "?")), 0) + 1
        )
    for edge in graph.get("edges", []):
        edge_type = str(edge.get("type", "?"))
        report.edges_in[edge_type] = report.edges_in.get(edge_type, 0) + 1

    agent_id = graph.get("agent_id") or config.agent_id
    known_ids, controlled_ids = _agent_lens(graph, agent_id, nodes_by_id, report)

    # ── hosts ────────────────────────────────────────────────────────────────
    host_ip_by_node: Dict[str, IP] = {}
    known_hosts: Set[IP] = set()
    controlled_hosts: Set[IP] = set()
    member_networks = _member_networks(graph, nodes_by_id)
    sole_owner = _sole_address_owners(nodes_by_id)

    for node_id in sorted(known_ids["host"] | controlled_ids):
        node = nodes_by_id.get(node_id)
        if node is None:
            report.drops.append(Drop("host", node_id, "edge references a missing node"))
            continue
        attributes = node.get("attributes", {}) or {}
        label = str(node.get("label", ""))

        if float(node.get("confidence", 1.0)) < config.min_confidence:
            report.drops.append(Drop("host", node_id, "below min_confidence", label))
            continue

        claimed_elsewhere = frozenset(
            address
            for address in _routable_ipv4s(attributes.get("addresses") or ())
            if sole_owner.get(address, node_id) != node_id
        )
        address = _canonical_ipv4(
            attributes.get("addresses"),
            member_networks=member_networks.get(node_id, ()),
            claimed_elsewhere=claimed_elsewhere,
        )
        if address is None:
            # Hosts identified by DNS name and IPv6-only hosts reach this
            # branch. NetSecGame addresses hosts by IPv4 string, so there is no
            # representation for them.
            report.drops.append(Drop("host", node_id, "no routable IPv4 address", label))
            continue

        if not _in_scope(address, config):
            report.drops.append(Drop("host", node_id, "outside declared scope", label))
            continue

        host = IP(address)
        host_ip_by_node[node_id] = host
        known_hosts.add(host)
        if node_id in controlled_ids:
            controlled_hosts.add(host)
        if attributes.get("local"):
            provenance.local_hosts.add(host)
        if address in config.external_hosts or label in config.external_hosts:
            provenance.external_hosts.add(host)

        addresses = tuple(str(a) for a in (attributes.get("addresses") or []))
        provenance.host_addresses[host] = addresses
        provenance.node_id[host] = node_id
        provenance.confidence[host] = float(node.get("confidence", 1.0))
        routable = _routable_ipv4s(addresses)
        if len(routable) > 1:
            report.notes.append(
                f"host {node_id} has {len(routable)} routable IPv4 addresses "
                f"{routable}; projected as {address} only"
            )
        if claimed_elsewhere:
            # The same IP being both "me" and "a remote host" is not expressible
            # in NetSecGame: the surrogate could emit an action targeting itself.
            report.notes.append(
                f"host {node_id} also claims {sorted(claimed_elsewhere)}, which other "
                "host node(s) own; NetSecGame cannot express the aliasing"
            )

    if not config.scope_cidrs:
        off_range = sorted(
            str(host)
            for host in known_hosts
            if not ipaddress.IPv4Address(str(host)).is_private
        )
        if off_range:
            report.notes.append(
                f"no scope_cidrs set and {len(off_range)} of {len(known_hosts)} projected hosts "
                f"are public internet addresses ({', '.join(off_range[:4])}"
                f"{', ...' if len(off_range) > 4 else ''}); NetSecGame treats every known host as "
                "attackable, so declare a scope before any live run"
            )

    if config.treat_external_as_controlled:
        for designated in sorted(config.external_hosts):
            address = _canonical_ipv4([designated])
            if address is None:
                report.notes.append(
                    f"external host {designated!r} is not a routable IPv4 address; ignored"
                )
                continue
            host = IP(address)
            provenance.external_hosts.add(host)
            was_known = host in known_hosts
            known_hosts.add(host)
            controlled_hosts.add(host)
            report.notes.append(
                f"declared external host {address} added to controlled_hosts"
                + ("" if was_known else " (and to known_hosts: absent from the graph)")
                + " so ExfiltrateData has a destination; this is configuration, not observation"
            )

    # NetSecGame requires controlled_hosts to be a subset of known_hosts.
    known_hosts |= controlled_hosts
    if len(controlled_hosts) < 2:
        report.notes.append(
            f"{len(controlled_hosts)} controlled host(s): ExfiltrateData needs a destination "
            "different from the data's host, so exfiltration is unreachable"
        )
    if not controlled_hosts:
        report.notes.append(
            "no CONTROLS edge projected: without a controlled host the NetSecGame action "
            "space is empty, since every action needs a source_host"
        )

    # ── networks ─────────────────────────────────────────────────────────────
    known_networks: Set[Network] = set()
    for node_id in sorted(known_ids["network"]):
        node = nodes_by_id.get(node_id)
        if node is None:
            report.drops.append(Drop("network", node_id, "edge references a missing node"))
            continue
        attributes = node.get("attributes", {}) or {}
        label = str(node.get("label", ""))
        cidr = str(attributes.get("cidr", "") or "")

        if cidr in config.excluded_cidrs:
            report.drops.append(Drop("network", node_id, "excluded cidr", label))
            continue
        if attributes.get("inferred") and not config.include_inferred_networks:
            report.drops.append(Drop("network", node_id, "inferred network", label))
            continue
        if float(node.get("confidence", 1.0)) < config.min_confidence:
            report.drops.append(Drop("network", node_id, "below min_confidence", label))
            continue

        parsed = _projectable_network(cidr)
        if parsed is None:
            report.drops.append(Drop("network", node_id, "not a projectable IPv4 network", label))
            continue
        if config.scope_cidrs and not any(
            parsed.subnet_of(scope)
            for scope in _parsed_scope(config)
        ):
            report.drops.append(Drop("network", node_id, "outside declared scope", label))
            continue

        network = Network(str(parsed.network_address), parsed.prefixlen)
        known_networks.add(network)
        provenance.node_id[network] = node_id
        provenance.confidence[network] = float(node.get("confidence", 1.0))

    # ── services ─────────────────────────────────────────────────────────────
    known_services: Dict[IP, Set[Service]] = defaultdict(set)
    for node_id in sorted(known_ids["service"]):
        node = nodes_by_id.get(node_id)
        if node is None:
            report.drops.append(Drop("service", node_id, "edge references a missing node"))
            continue
        attributes = node.get("attributes", {}) or {}
        label = str(node.get("label", ""))

        if float(node.get("confidence", 1.0)) < config.min_confidence:
            report.drops.append(Drop("service", node_id, "below min_confidence", label))
            continue

        statuses = _status_set(attributes.get("status"))
        if (statuses & config.service_status_denylist) and not (
            statuses & config.service_status_positive
        ):
            report.drops.append(Drop("service", node_id, f"status={sorted(statuses)}", label))
            continue

        host = host_ip_by_node.get(str(attributes.get("host_id", "")))
        if host is None:
            report.drops.append(Drop("service", node_id, "host not projected", label))
            continue

        service = Service(
            name=_service_name(attributes),
            type=str(attributes.get("protocol", "unknown")),
            version="unknown",
            is_local=bool(host in provenance.local_hosts),
        )
        known_services[host].add(service)
        provenance.node_id[(host, service)] = node_id
        provenance.confidence[(host, service)] = float(node.get("confidence", 1.0))
        port = attributes.get("port")
        if isinstance(port, int):
            provenance.service_ports[(host, service)] = port

    # ── data ─────────────────────────────────────────────────────────────────
    known_data: Dict[IP, Set[Data]] = defaultdict(set)
    denylist = tuple(re.compile(pattern) for pattern in config.data_path_denylist)
    data_nodes = [
        (node_id, nodes_by_id[node_id])
        for node_id in sorted(known_ids["data"])
        if node_id in nodes_by_id
    ]
    # Highest confidence first, so a per-host cap keeps the best-supported items.
    data_nodes.sort(key=lambda pair: (-float(pair[1].get("confidence", 1.0)), pair[0]))

    for node_id, node in data_nodes:
        attributes = node.get("attributes", {}) or {}
        label = str(node.get("label", ""))
        locator = str(attributes.get("locator", "") or "")

        if float(node.get("confidence", 1.0)) < config.min_confidence:
            report.drops.append(Drop("data", node_id, "below min_confidence", label))
            continue
        if not locator:
            report.drops.append(Drop("data", node_id, "no locator", label))
            continue
        present, presence_reason = _data_is_present(
            attributes, accept_command_argument=config.accept_command_argument_data
        )
        if config.require_confirmed_data and not present:
            report.drops.append(Drop("data", node_id, presence_reason, label))
            continue
        if any(pattern.search(locator) for pattern in denylist):
            report.drops.append(Drop("data", node_id, "denylisted path", label))
            continue

        host = host_ip_by_node.get(str(attributes.get("host_id", "")))
        if host is None:
            report.drops.append(Drop("data", node_id, "host not projected", label))
            continue
        if len(known_data[host]) >= config.max_data_per_host:
            report.drops.append(Drop("data", node_id, "over max_data_per_host", label))
            continue

        datapoint = Data(owner=str(host), id=locator, size=0, type="file", content=locator)
        known_data[host].add(datapoint)
        provenance.node_id[datapoint] = node_id
        provenance.confidence[datapoint] = float(node.get("confidence", 1.0))

    # ── blocks ───────────────────────────────────────────────────────────────
    blocks = []
    for node_id in sorted(known_ids["block"]):
        node = nodes_by_id.get(node_id)
        if node is not None:
            blocks.append(node)
    provenance.blocks = tuple(blocks)
    if blocks:
        report.notes.append(
            f"{len(blocks)} block node(s) not projected: they are attributed to a target "
            "service with no observing source host, while NetSecGame needs Dict[IP, Set[IP]]"
        )

    present_unrepresentable = sorted(
        node_type
        for node_type in UNREPRESENTABLE_NODE_TYPES - {"agent", "observation"}
        if report.nodes_in.get(node_type)
    )
    if present_unrepresentable:
        detail = ", ".join(f"{t}={report.nodes_in[t]}" for t in present_unrepresentable)
        report.notes.append(f"node types with no GameState representation dropped: {detail}")

    state = GameState(
        controlled_hosts=controlled_hosts,
        known_hosts=known_hosts,
        known_services={host: services for host, services in known_services.items() if services},
        known_data={host: items for host, items in known_data.items() if items},
        known_networks=known_networks,
        known_blocks={},
    )
    report.counts_out = {
        "known_networks": len(state.known_networks),
        "known_hosts": len(state.known_hosts),
        "controlled_hosts": len(state.controlled_hosts),
        "known_services": sum(len(v) for v in state.known_services.values()),
        "known_data": sum(len(v) for v in state.known_data.values()),
        "known_blocks": 0,
    }
    return Projection(state=state, report=report, provenance=provenance)


def project_path(
    path: str, config: Optional[AdapterConfig] = None
) -> Projection:
    """Convenience wrapper: locate a graph on disk and project it."""
    graph, source = load_graph(path)
    return project_graph_to_game_state(graph, config=config, source=source)


# ── helpers ─────────────────────────────────────────────────────────────────


def _agent_lens(
    graph: Dict[str, Any],
    agent_id: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    report: AdapterReport,
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Collect the node ids the agent knows and controls.

    Falls back to every node of a projectable type when the graph carries no
    knowledge edges, which keeps the adapter usable on hand-built fixtures.
    """
    known: Dict[str, Set[str]] = {category: set() for category in KNOWLEDGE_EDGES.values()}
    controlled: Set[str] = set()
    saw_knowledge_edge = False

    for edge in graph.get("edges", []):
        edge_type = str(edge.get("type", ""))
        if edge.get("source") != agent_id:
            continue
        target = edge.get("target")
        if not target:
            continue
        if edge_type in KNOWLEDGE_EDGES:
            known[KNOWLEDGE_EDGES[edge_type]].add(target)
            saw_knowledge_edge = True
        elif edge_type == CONTROL_EDGE:
            controlled.add(target)

    if not saw_knowledge_edge:
        report.notes.append(
            f"no KNOWS_* edge from {agent_id!r}: falling back to every node of a "
            "projectable type, which assumes the whole graph is agent knowledge"
        )
        for node_id, node in nodes_by_id.items():
            category = str(node.get("type", ""))
            if category in known:
                known[category].add(node_id)

    if not controlled:
        # The local container is always controlled; the CONTROLS edge is the
        # normal carrier of that fact, `local: true` the fallback.
        for node_id, node in nodes_by_id.items():
            if node.get("type") == "host" and (node.get("attributes") or {}).get("local"):
                controlled.add(node_id)
                report.notes.append(
                    f"no CONTROLS edge from {agent_id!r}: treating local host {node_id} as controlled"
                )

    known["host"] |= controlled
    return known, controlled


def _parsed_scope(config: AdapterConfig) -> Tuple[ipaddress.IPv4Network, ...]:
    scope: List[ipaddress.IPv4Network] = []
    for cidr in sorted(config.scope_cidrs):
        try:
            scope.append(ipaddress.IPv4Network(cidr, strict=False))
        except ValueError:
            continue
    return tuple(scope)


def _in_scope(address: str, config: AdapterConfig) -> bool:
    """Whether this host is a declared target rather than merely contacted."""
    if not config.scope_cidrs:
        return True
    if address in config.external_hosts:
        return True
    parsed = ipaddress.IPv4Address(address)
    return any(parsed in scope for scope in _parsed_scope(config))


def _member_networks(
    graph: Dict[str, Any], nodes_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Tuple[str, ...]]:
    """host node id -> CIDRs it is a MEMBER_OF, used to disambiguate addresses."""
    out: Dict[str, List[str]] = defaultdict(list)
    for edge in graph.get("edges", []):
        if str(edge.get("type", "")) != "MEMBER_OF":
            continue
        network = nodes_by_id.get(str(edge.get("target", "")))
        if network is None or network.get("type") != "network":
            continue
        cidr = str((network.get("attributes") or {}).get("cidr", "") or "")
        if cidr:
            out[str(edge.get("source", ""))].append(cidr)
    return {node_id: tuple(sorted(cidrs)) for node_id, cidrs in out.items()}


def _sole_address_owners(nodes_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Map each address to the host node whose only routable address it is.

    A node with exactly one routable IPv4 address is identified by it. When a
    multi-homed node also lists that address, as a container does for its bridge
    gateways, the single-address node is its owner.
    """
    owners: Dict[str, str] = {}
    for node_id, node in sorted(nodes_by_id.items()):
        if node.get("type") != "host":
            continue
        addresses = _routable_ipv4s((node.get("attributes") or {}).get("addresses") or ())
        if len(addresses) == 1:
            owners.setdefault(addresses[0], node_id)
    return owners


def _status_set(status: Any) -> Set[str]:
    """Normalise a node's `status`, which may be a string or a list."""
    if status is None:
        return set()
    values = status if isinstance(status, (list, tuple)) else [status]
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _data_is_present(
    attributes: Dict[str, Any], accept_command_argument: bool = True
) -> Tuple[bool, str]:
    """Decide whether a data node belongs in NetSecGame's `known_data`.

    The state creator reports existence two ways and neither is a plain boolean:
    `exists` is `true`, or `[true, false]` when the file was observed both
    present and absent (created then deleted), and `existence: "unknown"`
    appears on nodes it could not reconcile.

    Filesystem confirmation is not the only evidence that matters, though.
    NetSecGame's `known_data` means *the agent has discovered this data*, and a file
    named on the operator's command line proves exactly that — `/etc/passwd` in
    the strategic sample run arrives with `knowledge_source:
    "command-argument"`, `existence: "unknown"`, sourced from the TTY log,
    because the file monitor never reconciled `/etc`. Requiring filesystem
    confirmation would drop the one attack-relevant file in the run while
    keeping two debconf caches that inotify happened to see.
    """
    existence = attributes.get("existence")
    if existence == "confirmed":
        return True, "confirmed"

    exists = attributes.get("exists")
    if exists is True:
        return True, "exists"
    if isinstance(exists, (list, tuple)):
        values = {bool(value) for value in exists}
        if values == {True}:
            return True, "exists"
        return False, "existence unstable (observed present and absent)"
    if exists is False:
        return False, "observed absent"

    if accept_command_argument and "command-argument" in _status_set(
        attributes.get("knowledge_source")
    ):
        return True, "named in a command"

    if existence is not None:
        return False, f"existence={existence!r}"
    return False, "existence not reported"


def _routable_ipv4s(addresses: Iterable[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for candidate in addresses or ():
        try:
            parsed = ipaddress.ip_address(str(candidate))
        except ValueError:
            continue
        if not isinstance(parsed, ipaddress.IPv4Address):
            continue
        if (
            parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_multicast
            or parsed.is_unspecified
            or parsed.is_reserved
        ):
            continue
        out.append(str(parsed))
    return tuple(out)


def _canonical_ipv4(
    addresses: Optional[Sequence[str]],
    member_networks: Sequence[str] = (),
    claimed_elsewhere: FrozenSet[str] = frozenset(),
) -> Optional[str]:
    """Pick the one address NetSecGame will use to identify a host.

    NetSecGame identifies a host by a single IPv4 string, but the observed container is
    multi-homed: `host:local` in the sample run claims 127.0.0.1, 172.23.0.2,
    172.17.0.2, 172.17.0.1, 172.23.0.1, several link-local v6 addresses and a
    multicast address. Taking the numerically smallest routable one picks
    172.17.0.1 — a docker bridge *gateway*, which is a different machine and is
    separately present in the graph as its own host node.

    So the choice is ranked, not arbitrary:
      1. addresses inside a network this host is a MEMBER_OF (its own subnets);
      2. addresses no other host node claims (excludes gateways and aliases);
      3. numerically smallest, for stability across rebuilds.
    """
    routable = _routable_ipv4s(addresses or ())
    if not routable:
        return None

    networks: List[ipaddress.IPv4Network] = []
    for cidr in member_networks:
        parsed = _projectable_network(str(cidr))
        if parsed is not None:
            networks.append(parsed)

    def rank(address: str) -> Tuple[int, int, int]:
        parsed = ipaddress.IPv4Address(address)
        in_member_network = any(parsed in network for network in networks)
        return (
            0 if in_member_network else 1,
            1 if address in claimed_elsewhere else 0,
            int(parsed),
        )

    return min(routable, key=rank)


def _projectable_network(cidr: str) -> Optional[ipaddress.IPv4Network]:
    if not cidr:
        return None
    try:
        network = ipaddress.IPv4Network(cidr, strict=False)
    except ValueError:
        return None
    if (
        network.is_loopback
        or network.is_link_local
        or network.is_multicast
        or network.is_unspecified
    ):
        return None
    if network.prefixlen == 32:
        # The state creator documents /32 as a host, never a network.
        return None
    return network


def _service_name(attributes: Dict[str, Any]) -> str:
    """Collapse `service_name` into NetSecGame's single-string service name.

    The state creator emits either a string or a list of aliases
    (`["https", "ssl"]`). Stringifying a list produces a nonsense NetSecGame service
    name, so pick the first alias and keep the ordering deterministic.
    """
    raw = attributes.get("service_name")
    if isinstance(raw, str) and raw:
        return raw
    if isinstance(raw, (list, tuple)) and raw:
        return str(raw[0])
    port = attributes.get("port")
    protocol = attributes.get("protocol", "unknown")
    if isinstance(port, int):
        return f"{protocol}/{port}"
    return "unknown"
