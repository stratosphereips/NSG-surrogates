"""Command line entry point: `python -m nsg_surrogate <command>`.

    inspect   project a docker state graph and report what the mapping loses
    state     print the projected GameState as NSG JSON
    act       choose one parameterized NSG action for a state graph (needs torch)

`inspect` and `state` need only `netsecgame` on the path; `act` additionally
needs torch and torch-geometric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

from netsecgame.game_components import ActionType

from . import candidates as candidates_mod
from .state_adapter import AdapterConfig, Projection, project_path


def _adapter_config(args: argparse.Namespace) -> AdapterConfig:
    return AdapterConfig(
        min_confidence=args.min_confidence,
        include_inferred_networks=args.include_inferred_networks,
        require_confirmed_data=not args.unconfirmed_data,
        data_path_denylist=() if args.no_data_denylist else AdapterConfig.data_path_denylist,
        max_data_per_host=args.max_data_per_host,
        service_status_denylist=(
            frozenset() if args.any_service_status else AdapterConfig.service_status_denylist
        ),
        scope_cidrs=frozenset(args.scope or ()),
        external_hosts=frozenset(args.external_host or ()),
    )


def _add_adapter_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="observation root, state dir, or graph.json")
    parser.add_argument("--min-confidence", type=float, default=AdapterConfig.min_confidence)
    parser.add_argument("--include-inferred-networks", action="store_true")
    parser.add_argument(
        "--unconfirmed-data",
        action="store_true",
        help="keep data whose existence the observer could not confirm",
    )
    parser.add_argument("--no-data-denylist", action="store_true", help="keep OS/container noise paths")
    parser.add_argument(
        "--any-service-status",
        action="store_true",
        help="keep services the observer only saw attempted (our own scan traffic)",
    )
    parser.add_argument("--max-data-per-host", type=int, default=AdapterConfig.max_data_per_host)
    parser.add_argument(
        "--scope",
        action="append",
        metavar="CIDR",
        help="network that is in scope as a target; repeatable. Without it, every host "
        "the container ever talked to becomes an NSG target, including the public internet",
    )
    parser.add_argument(
        "--external-host",
        action="append",
        metavar="IP",
        help="host designated as outside the range (exfiltration destination); repeatable",
    )


def _report_dict(projection: Projection, actions: List) -> Dict[str, object]:
    report = projection.report
    return {
        "source": report.source,
        "schema_version": report.schema_version,
        "detail_level": report.detail_level,
        "nodes_in": report.nodes_in,
        "counts_out": report.counts_out,
        "dropped": report.dropped_by_reason,
        "notes": report.notes,
        "action_space": candidates_mod.breakdown(actions),
        "action_space_total": len(actions),
        "action_space_executable": len(candidates_mod.executable(actions)),
    }


def cmd_inspect(args: argparse.Namespace) -> int:
    projection = project_path(args.path, config=_adapter_config(args))
    actions = candidates_mod.enumerate_actions(projection.state)

    if args.json:
        print(json.dumps(_report_dict(projection, actions), indent=2, sort_keys=True))
        return 0

    report = projection.report
    print(f"source          {report.source}")
    print(f"graph schema    {report.schema_version or '<unset>'}  level={report.detail_level or '<unset>'}")
    print()
    print("nodes in graph")
    for node_type, count in sorted(report.nodes_in.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {node_type:<12} {count:>5}")
    print()
    print("projected GameState")
    for category, count in report.counts_out.items():
        print(f"  {category:<17} {count:>5}")
    print()

    dropped = report.dropped_by_reason
    if dropped:
        print(f"dropped ({len(report.drops)} entities)")
        for reason, count in sorted(dropped.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {count:>5}  {reason}")
        print()

    print(f"action space ({len(actions)} candidates)")
    for action_type, count in candidates_mod.breakdown(actions).items():
        status = candidates_mod.TRANSLATOR_SUPPORT.get(ActionType[action_type], "unknown")
        print(f"  {action_type:<16} {count:>5}   translator: {status}")
    executable = candidates_mod.executable(actions)
    share = (100.0 * len(executable) / len(actions)) if actions else 0.0
    print(f"  {'executable now':<16} {len(executable):>5}   ({share:.0f}% of candidates)")

    if report.notes:
        print()
        print("notes")
        for note in report.notes:
            print(f"  - {note}")

    if args.show_drops:
        print()
        print(f"first {args.show_drops} dropped entities")
        for drop in report.drops[: args.show_drops]:
            print(f"  - {drop}")
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    projection = project_path(args.path, config=_adapter_config(args))
    print(projection.state.as_json())
    return 0


def cmd_act(args: argparse.Namespace) -> int:
    projection = project_path(args.path, config=_adapter_config(args))
    actions = candidates_mod.enumerate_actions(projection.state)
    if not actions:
        print("no valid action for this state", file=sys.stderr)
        return 2

    from .policy import SurrogatePolicy  # imported lazily: needs torch

    policy = SurrogatePolicy.load(args.weights)
    decision = policy.act(
        projection.state,
        actions,
        provenance=projection.provenance,
        temperature=args.temperature,
    )

    if args.json:
        print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
    else:
        print(candidates_mod.describe(decision.action))
        print(f"  translator support: {candidates_mod.support_status(decision.action)}")
        print(f"  candidates: {len(actions)}")
        for head, choice in decision.head_choices.items():
            print(f"  {head}: {choice}")
        print(decision.action.to_json())
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from . import dataset as dataset_mod

    result = dataset_mod.build(
        args.path,
        config=_adapter_config(args),
        max_records_per_transition=args.max_records_per_transition or None,
    )
    report = result.report

    if args.out:
        paths = dataset_mod.write(
            result, args.out, dataset_mod._trajectory_dir(args.path)
        )
        for name, path in paths.items():
            print(f"wrote {name:<11} {path}")
        print(f"wrote graphs      {os.path.join(args.out, 'graphs')}/")
        print()

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0

    print(f"run              {report.run_id}")
    print(f"states           {report.states}   transitions: {report.transitions}")
    print(
        f"action records   {report.action_records}"
        f"  -> {report.actions_after_dedupe} after dedupe"
    )
    for source, count in sorted(report.action_sources.items(), key=lambda kv: -kv[1]):
        print(f"  {source:<22} {count:>7}")
    print()

    print(f"labels           {report.labels}")
    for source, count in sorted(report.labels_by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {source:<22} {count:>7}")
    print()
    print("labels by action type")
    for action_type, count in report.labels_by_type.items():
        print(f"  {action_type:<22} {count:>7}   translator: "
              f"{candidates_mod.TRANSLATOR_SUPPORT.get(ActionType[action_type], '?')}")
    print()
    print(
        f"transitions with no label   {report.transitions_with_no_label} of {report.transitions}"
        f"   (NSG-visible diff empty in {report.transitions_with_empty_diff})"
    )
    if report.knowledge_lost:
        print(f"knowledge lost (unrepresentable in NSG): {report.knowledge_lost}")
    print()

    if report.unmappable:
        print("cannot be expressed in NetSecGame today")
        for category, count in report.unmappable.items():
            print(f"  {count:>7}  {category}")
        print()
    if report.rejected:
        print("labels rejected by the candidate gate")
        for reason, count in report.rejected.items():
            print(f"  {count:>7}  {reason}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import training

    config = AdapterConfig(
        min_confidence=args.min_confidence,
        scope_cidrs=frozenset(args.scope or ()),
        external_hosts=frozenset(args.external_host or ()),
    )
    samples, load_report = training.load_samples(
        args.datasets, config=config, weight_by_confidence=not args.unweighted
    )
    print(f"loaded {load_report.samples} samples from {load_report.rows} rows")
    print(f"  by type   {load_report.by_type}")
    print(f"  by source {load_report.by_source}")
    print(f"  by run    {load_report.runs}")
    if load_report.skipped:
        print(f"  skipped   {load_report.skipped}")
    if load_report.projection_mismatches:
        print(
            f"  WARNING: {load_report.projection_mismatches} row(s) re-project differently "
            "than their cached projection — the adapter changed since the dataset was built"
        )
    if not samples:
        print("nothing trainable", file=sys.stderr)
        return 2

    train_samples, holdout = training.split_by_run(samples, args.holdout_run or ())
    print(f"train {len(train_samples)}  holdout {len(holdout)}")
    if not holdout:
        print("  (no holdout run named: evaluation below is in-sample)")
    print()

    from .policy import FactoredGNNPolicy

    policy = FactoredGNNPolicy()
    result = training.train(
        train_samples,
        holdout=holdout,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        policy=policy,
        load=load_report,
        verbose=not args.quiet,
    )

    print()
    print(f"train  {result.train_eval.as_dict()}")
    if holdout:
        print(f"holdout {result.holdout_eval.as_dict()}")
    evaluated = result.holdout_eval if holdout else result.train_eval
    print(
        f"majority-class baseline {evaluated.majority_type_baseline:.2f}"
        " — accuracy at or below this is noise"
    )
    print(
        f"deterministic ceiling   {evaluated.deterministic_ceiling:.2f}"
        f" over {evaluated.distinct_inputs} distinct encoder inputs"
        " — identical inputs with different labels cannot be fitted"
    )

    if args.out:
        result.weights_path = training.save(policy, args.out)
        print(f"weights -> {result.weights_path}")
    if args.metrics:
        with open(args.metrics, "w", encoding="utf-8") as handle:
            json.dump(result.as_dict(), handle, indent=2, sort_keys=True)
        print(f"metrics -> {args.metrics}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nsg_surrogate", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="report the docker -> NSG mapping")
    _add_adapter_flags(inspect)
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("--show-drops", type=int, default=0, metavar="N")
    inspect.set_defaults(func=cmd_inspect)

    state = subparsers.add_parser("state", help="print the projected GameState as JSON")
    _add_adapter_flags(state)
    state.set_defaults(func=cmd_state)

    dataset = subparsers.add_parser(
        "dataset", help="build (state, action) pairs from a trajectory and report the yield"
    )
    _add_adapter_flags(dataset)
    dataset.add_argument("--out", default=None, metavar="DIR", help="write the dataset here")
    dataset.add_argument(
        "--max-records-per-transition",
        type=int,
        default=200,
        help="cap action records loaded per transition (counts stay exact; 0 for no cap)",
    )
    dataset.add_argument("--json", action="store_true")
    dataset.set_defaults(func=cmd_dataset)

    train = subparsers.add_parser(
        "train", help="behaviour-clone the surrogate on labelled pairs (needs torch)"
    )
    train.add_argument("datasets", nargs="+", help="dataset directories holding pairs.jsonl")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--holdout-run", action="append", metavar="RUN_ID",
        help="run to keep out of training; repeatable. Splits are by run because "
        "rows share states, and a row-level split leaks",
    )
    train.add_argument("--min-confidence", type=float, default=AdapterConfig.min_confidence)
    train.add_argument("--scope", action="append", metavar="CIDR")
    train.add_argument("--external-host", action="append", metavar="IP")
    train.add_argument(
        "--unweighted", action="store_true", help="ignore label confidence as a sample weight"
    )
    train.add_argument("--out", default=None, metavar="PATH", help="write the checkpoint here")
    train.add_argument("--metrics", default=None, metavar="PATH", help="write metrics JSON here")
    train.add_argument("--quiet", action="store_true")
    train.set_defaults(func=cmd_train)

    act = subparsers.add_parser("act", help="choose one NSG action (needs torch)")
    _add_adapter_flags(act)
    act.add_argument("--weights", default=None, help="policy checkpoint; random init when omitted")
    act.add_argument("--temperature", type=float, default=0.1)
    act.add_argument("--json", action="store_true")
    act.set_defaults(func=cmd_act)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
