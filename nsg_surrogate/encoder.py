"""`GameState` -> PyTorch Geometric `HeteroData`.

Feature layout, node types, edge types and node ordering are byte-compatible
with `sgrl_netsec/policy_netsec.py:state_to_pyg`, so a checkpoint trained in the
simulator loads and means the same thing here. Two feature *sources* differ,
because the simulator's versions are unusable in a container range:

`host[1]` "is public"
    The simulator reads `netaddr.IPAddress(...).is_private()`. A docker range is
    entirely RFC1918, so that feature is 0 for every host and the exfiltration
    destination head has nothing to discriminate on. Here the bit means "is a
    designated external host", supplied by `Provenance.external_hosts`.

`service[0]` "port"
    The simulator parses `int(service.name.split("/")[0])`, but `Service.name`
    holds a service *name* (`ssh`, `https`), so the parse raises and the feature
    is silently always 0 — a dead input in the trained policy. The state creator
    reports the real port, carried here in `Provenance.service_ports`.

Both are noted rather than hidden: a simulator-trained checkpoint has never seen
a nonzero `service[0]`, so enabling it changes the input distribution. Pass
`legacy_service_port=True` to reproduce the simulator's dead feature exactly
when comparing against simulator behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import netaddr
import torch
from torch_geometric.data import HeteroData

from netsecgame.game_components import Data, GameState, IP, Network, Service

from .attempt_counts import AttemptCounts
from .state_adapter import Provenance

FEATURE_DIM = 16
NODE_TYPES = ("network", "host", "service", "data")
EDGE_TYPES = (
    ("host", "in", "network"),
    ("network", "contains", "host"),
    ("host", "runs", "service"),
    ("host", "stores", "data"),
    ("service", "rev_runs", "host"),
    ("data", "rev_stores", "host"),
)
#: Attempt counters and the summary vector are clipped here before scaling.
COUNT_CLIP = 10.0


def _norm(count: float) -> float:
    return min(float(count), COUNT_CLIP) / COUNT_CLIP


def _is_external(host: IP, provenance: Optional[Provenance]) -> bool:
    """Designated-external test, replacing the simulator's public-IP test."""
    if provenance is not None and provenance.external_hosts:
        return host in provenance.external_hosts
    try:
        return not netaddr.IPAddress(str(host)).is_private()
    except Exception:
        return False


def state_to_pyg(
    state: GameState,
    attempt_counts: Optional[AttemptCounts] = None,
    provenance: Optional[Provenance] = None,
    legacy_service_port: bool = False,
) -> Tuple[HeteroData, Dict[str, dict], Dict[str, list]]:
    """Build the heterogeneous graph plus both index maps.

    Node order within each type is `sorted(..., key=str)` — set iteration order
    is random in Python, and a scrambled node order would scramble the topology
    the policy sees between two calls on the same state.
    """
    data = HeteroData()
    object_to_idx: Dict[str, dict] = {node_type: {} for node_type in NODE_TYPES}

    # ── networks ─────────────────────────────────────────────────────────────
    networks: List[Network] = sorted(state.known_networks, key=str)
    for idx, network in enumerate(networks):
        object_to_idx["network"][network] = idx

    all_known_hosts = set(state.known_hosts) | set(state.controlled_hosts)
    network_host_counts: Dict[Network, int] = {}
    for network in networks:
        try:
            cidr = netaddr.IPNetwork(str(network))
            network_host_counts[network] = sum(1 for h in all_known_hosts if str(h) in cidr)
        except netaddr.AddrFormatError:
            network_host_counts[network] = 0

    network_features = []
    for network in networks:
        feature = torch.zeros(FEATURE_DIM)
        if attempt_counts is not None:
            feature[0] = _norm(attempt_counts.scan.get(network, 0))
        host_count = network_host_counts[network]
        feature[1] = 1.0 if host_count > 0 else 0.0
        feature[2] = _norm(host_count)
        network_features.append(feature)
    data["network"].x = (
        torch.stack(network_features) if network_features else torch.empty((0, FEATURE_DIM))
    )

    # ── hosts ────────────────────────────────────────────────────────────────
    hosts: List[IP] = sorted(all_known_hosts, key=str)
    for idx, host in enumerate(hosts):
        object_to_idx["host"][host] = idx

    host_features = []
    for host in hosts:
        feature = torch.zeros(FEATURE_DIM)
        if host in state.controlled_hosts:
            feature[0] = 1.0
        if _is_external(host, provenance):
            feature[1] = 1.0
        if attempt_counts is not None:
            feature[2] = _norm(attempt_counts.findservices.get(host, 0))
            feature[3] = _norm(attempt_counts.finddata.get(host, 0))
            feature[4] = _norm(attempt_counts.exploit.get(host, 0))
        host_features.append(feature)
    data["host"].x = torch.stack(host_features) if host_features else torch.empty((0, FEATURE_DIM))

    # ── services: one node per (host, service) pair ───────────────────────────
    services: List[Tuple[IP, Service]] = []
    for host in sorted(state.known_services.keys(), key=str):
        for service in sorted(state.known_services[host], key=str):
            object_to_idx["service"][(host, service)] = len(services)
            services.append((host, service))

    service_features = []
    for host, service in services:
        feature = torch.zeros(FEATURE_DIM)
        port = _service_port(host, service, provenance, legacy_service_port)
        if port is not None:
            feature[0] = min(max(port, 0), 65535) / 65535.0
        feature[1] = 1.0 if getattr(service, "is_local", False) else 0.0
        service_features.append(feature)
    data["service"].x = (
        torch.stack(service_features) if service_features else torch.empty((0, FEATURE_DIM))
    )

    # ── data ─────────────────────────────────────────────────────────────────
    # "Exfiltrated" means: a copy sits on a controlled host that is outside the
    # range. In the simulator that was "controlled and public".
    external_controlled = {
        host for host in state.controlled_hosts if _is_external(host, provenance)
    }
    exfiltrated = {
        datapoint
        for host in external_controlled
        for datapoint in state.known_data.get(host, ())
    }

    datapoints: List[Tuple[Data, IP]] = []
    for host in sorted(state.known_data.keys(), key=str):
        for datapoint in sorted(state.known_data[host], key=str):
            if datapoint not in object_to_idx["data"]:
                object_to_idx["data"][datapoint] = len(datapoints)
                datapoints.append((datapoint, host))

    data_features = []
    for datapoint, host in datapoints:
        feature = torch.zeros(FEATURE_DIM)
        feature[0] = 1.0 if datapoint in exfiltrated else 0.0
        feature[1] = 1.0 if _is_external(host, provenance) else 0.0
        if attempt_counts is not None:
            feature[2] = _norm(attempt_counts.exfil.get(datapoint, 0))
        data_features.append(feature)
    data["data"].x = torch.stack(data_features) if data_features else torch.empty((0, FEATURE_DIM))

    # ── edges ────────────────────────────────────────────────────────────────
    host_network_edges: List[List[int]] = []
    for host_idx, host in enumerate(hosts):
        for network_idx, network in enumerate(networks):
            try:
                if str(host) in netaddr.IPNetwork(str(network)):
                    host_network_edges.append([host_idx, network_idx])
            except netaddr.AddrFormatError:
                pass
    _set_edges(data, ("host", "in", "network"), host_network_edges)
    _set_edges(data, ("network", "contains", "host"), _reverse(host_network_edges))

    host_service_edges = [
        [object_to_idx["host"][host], service_idx]
        for service_idx, (host, _) in enumerate(services)
        if host in object_to_idx["host"]
    ]
    _set_edges(data, ("host", "runs", "service"), host_service_edges)
    _set_edges(data, ("service", "rev_runs", "host"), _reverse(host_service_edges))

    host_data_edges = [
        [object_to_idx["host"][host], data_idx]
        for data_idx, (_, host) in enumerate(datapoints)
        if host in object_to_idx["host"]
    ]
    _set_edges(data, ("host", "stores", "data"), host_data_edges)
    _set_edges(data, ("data", "rev_stores", "host"), _reverse(host_data_edges))

    idx_to_object = {
        "network": list(networks),
        "host": list(hosts),
        "service": list(services),
        "data": [datapoint for datapoint, _ in datapoints],
    }
    return data, object_to_idx, idx_to_object


def state_summary(state: GameState, provenance: Optional[Provenance] = None) -> torch.Tensor:
    """Phase-of-attack vector fed alongside the pooled graph embedding.

      [0] controlled hosts with no known data      -> need FindData
      [1] known data not yet exfiltrated           -> need ExfiltrateData
      [2] uncontrolled hosts with known services   -> can ExploitService

    Kept dimension- and order-identical to the simulator agent's
    `_state_summary`. Note the middle term counts data on hosts the agent does
    not control, which in the real range is 0 whenever every data item was found
    on the observed container itself.
    """
    controlled = state.controlled_hosts
    controlled_without_data = sum(1 for host in controlled if not state.known_data.get(host))
    data_not_exfiltrated = sum(
        len(items) for host, items in state.known_data.items() if host not in controlled
    )
    exploitable = sum(
        1
        for host, services in state.known_services.items()
        if host not in controlled and services
    )
    return torch.tensor(
        [_norm(controlled_without_data), _norm(data_not_exfiltrated), _norm(exploitable)],
        dtype=torch.float32,
    )


def _service_port(
    host: IP,
    service: Service,
    provenance: Optional[Provenance],
    legacy: bool,
) -> Optional[int]:
    if legacy:
        # Reproduce the simulator's parse, which fails for named services.
        try:
            return int(service.name.split("/")[0])
        except (ValueError, IndexError):
            return None
    if provenance is not None:
        port = provenance.service_ports.get((host, service))
        if port is not None:
            return int(port)
    try:
        return int(service.name.split("/")[0])
    except (ValueError, IndexError):
        return None


def _reverse(edges: List[List[int]]) -> List[List[int]]:
    return [[dst, src] for src, dst in edges]


def _set_edges(data: HeteroData, edge_type: Tuple[str, str, str], edges: List[List[int]]) -> None:
    data[edge_type].edge_index = (
        torch.tensor(edges, dtype=torch.long).t()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
