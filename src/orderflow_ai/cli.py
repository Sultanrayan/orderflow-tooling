"""Command line interface.

One entry point with four sub-commands::

    orderflow live     # analyse a live exchange feed
    orderflow collect  # record ticks for later replay
    orderflow replay   # analyse a recorded file
    orderflow backtest # paper trade a recorded file

Every command prints the analysis in a compact, human readable form and returns
a non-zero exit code on failure, so the CLI composes with shell scripting.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .backtest import BacktestEngine, SignalRule
from .core.analysis import Analysis
from .core.config import OrderFlowConfig
from .core.pipeline import OrderFlowPipeline
from .ingestion.hub import IngestionHub
from .output.formatter import PayloadBuilder
from .output.optimizer import dumps

__all__ = ["main"]

DEFAULT_CONFIG = "configs/default.yaml"
EXIT_OK = 0
EXIT_ERROR = 1


def _add_global_options(parser: argparse.ArgumentParser, suppress: bool) -> None:
    """Add the options accepted both before and after the sub-command.

    Args:
        parser: Parser to extend.
        suppress: Use ``SUPPRESS`` defaults on sub-parsers so that a value given
            before the sub-command is not overwritten by the sub-parser default.
    """
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS if suppress else DEFAULT_CONFIG,
        help="path to a YAML config file",
    )
    parser.add_argument(
        "--log-level",
        default=argparse.SUPPRESS if suppress else "INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="logging verbosity",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every sub-command."""
    parser = argparse.ArgumentParser(
        prog="orderflow",
        description="AI powered order flow analysis engine.",
    )
    _add_global_options(parser, suppress=False)

    common = argparse.ArgumentParser(add_help=False)
    _add_global_options(common, suppress=True)

    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser(
        "live", parents=[common], help="analyse a live exchange feed"
    )
    live.add_argument("--symbol", default=None, help="override the configured symbol")
    _add_analysis_options(live)

    replay = subparsers.add_parser(
        "replay", parents=[common], help="analyse a recorded tick file"
    )
    replay.add_argument("--file", required=True, help="recorded JSON lines file")
    replay.add_argument("--bars", type=int, default=None, help="stop after N bars")
    _add_analysis_options(replay)

    collect = subparsers.add_parser(
        "collect", parents=[common], help="record ticks from a live feed"
    )
    collect.add_argument("--symbol", default=None, help="override the configured symbol")
    collect.add_argument("--duration", type=int, default=3600, help="seconds to record")
    collect.add_argument("--output", default=None, help="destination JSON lines file")

    backtest = subparsers.add_parser(
        "backtest", parents=[common], help="paper trade a recorded file"
    )
    backtest.add_argument("--file", default=None, help="recorded JSON lines file")
    backtest.add_argument("--start", default=None, help="ISO date lower bound, inclusive")
    backtest.add_argument("--end", default=None, help="ISO date upper bound, exclusive")
    backtest.add_argument("--bars", type=int, default=None, help="stop after N bars")
    backtest.add_argument("--capital", type=float, default=10_000.0, help="starting notional")
    backtest.add_argument(
        "--min-confidence", type=float, default=75.0, help="minimum pattern confidence to trade"
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success and ``1`` when the command failed.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )

    try:
        config = OrderFlowConfig.load(args.config)
        if args.command == "live":
            return asyncio.run(_run_live(config, args))
        if args.command == "replay":
            return asyncio.run(_run_replay(config, args))
        if args.command == "collect":
            return asyncio.run(_run_collect(config, args))
        if args.command == "backtest":
            return asyncio.run(_run_backtest(config, args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    parser.error(f"unknown command {args.command!r}")
    return EXIT_ERROR  # pragma: no cover - argparse exits first


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
async def _run_live(config: OrderFlowConfig, args: argparse.Namespace) -> int:
    """Stream a live feed and print each closed bar analysis."""
    config = _with_overrides(config, symbol=args.symbol, mode="live")
    pipeline = OrderFlowPipeline(config)
    async for analysis in pipeline.stream(limit=args.max_bars):
        _print_analysis(analysis, args.payload)
    _print_dict(pipeline.diagnostics())
    return EXIT_OK


async def _run_replay(config: OrderFlowConfig, args: argparse.Namespace) -> int:
    """Analyse a recorded file."""
    config = _with_replay_path(config, args.file)
    pipeline = OrderFlowPipeline(config)
    bars = 0
    async for analysis in pipeline.stream(limit=args.bars):
        _print_analysis(analysis, args.payload)
        bars += 1
    if bars == 0:
        print("no bars closed during the replay", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


async def _run_collect(config: OrderFlowConfig, args: argparse.Namespace) -> int:
    """Record ticks from a live feed into a JSON lines file."""
    config = _with_overrides(
        config,
        symbol=args.symbol,
        record_path=args.output or config.pipeline.record_path or "data/ticks.jsonl",
        mode="live",
    )
    hub = IngestionHub(config)
    await hub.start()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + args.duration
    trades = 0
    try:
        async for _ in hub.trades():
            trades += 1
            if loop.time() >= deadline:
                break
    finally:
        await hub.aclose()

    print(f"recorded {trades} trades to {config.pipeline.record_path}")
    return EXIT_OK if trades else EXIT_ERROR


async def _run_backtest(config: OrderFlowConfig, args: argparse.Namespace) -> int:
    """Paper trade a recorded file and print the metrics.

    A file given with ``--file`` always switches to replay mode, and a config
    already in replay mode must point at a source. Any other mode (synthetic by
    default) is left alone, so the documented ``--start``/``--end`` invocation
    runs without recorded data.
    """
    if args.file or config.pipeline.mode == "replay":
        config = _with_replay_path(config, args.file)
    config = _with_overrides(config, start=args.start, end=args.end)
    engine = BacktestEngine(
        config,
        rule=SignalRule(min_confidence=args.min_confidence),
        capital=args.capital,
    )
    result = await engine.run(max_bars=args.bars)
    _print_dict(result.metrics.to_dict())
    print(result.summary())
    return EXIT_OK


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by the analysis commands."""
    parser.add_argument("--max-bars", type=int, default=None, help="stop after N bars")
    parser.add_argument("--payload", action="store_true", help="print the AI payload as JSON")


def _with_overrides(
    config: OrderFlowConfig,
    symbol: str | None = None,
    record_path: str | None = None,
    mode: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> OrderFlowConfig:
    """Return a copy of ``config`` with the selected fields replaced.

    Only non-``None`` arguments are applied, so optional command line values can
    be passed straight through.
    """
    exchange = (
        dataclasses.replace(config.exchange, symbol=symbol.upper())
        if symbol
        else config.exchange
    )
    changes = {
        key: value
        for key, value in (
            ("record_path", record_path),
            ("mode", mode),
            ("start", start),
            ("end", end),
        )
        if value
    }
    pipeline = dataclasses.replace(config.pipeline, **changes) if changes else config.pipeline
    return dataclasses.replace(config, exchange=exchange, pipeline=pipeline)


def _with_replay_path(config: OrderFlowConfig, file: str | None) -> OrderFlowConfig:
    """Point the pipeline at ``file`` and force replay mode."""
    pipeline = dataclasses.replace(config.pipeline, mode="replay")
    if file:
        pipeline = dataclasses.replace(pipeline, replay_path=str(Path(file)))
    if not pipeline.replay_path:
        raise ValueError("no replay file configured; pass --file")
    return dataclasses.replace(config, pipeline=pipeline)


def _print_analysis(analysis: Analysis, with_payload: bool) -> None:
    """Print one analysis as a single readable line plus optional payload."""
    patterns = ", ".join(
        f"{match.name}@{match.price:g}/{match.confidence:.0f}"
        for match in analysis.patterns.patterns
    ) or f"fallback {analysis.patterns.confidence:.0f}"
    print(
        f"{analysis.open_time:%H:%M} {analysis.symbol} {analysis.price:g} "
        f"conf={analysis.confidence:.0f} dir={analysis.direction} [{patterns}] "
        f"structure={analysis.context.structure.label} session={analysis.context.session.primary}"
    )
    if with_payload:
        print(dumps(_payload_dict(analysis), compact=False))


def _payload_dict(analysis: Analysis) -> dict[str, Any]:
    """Optimised payload of an analysis, built with a fresh payload builder."""
    return PayloadBuilder().build_optimized(analysis).payload


def _print_dict(payload: dict[str, Any]) -> None:
    """Print a mapping as indented JSON."""
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())