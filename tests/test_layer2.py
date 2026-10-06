"""Layer 2: the ten order flow primitives."""

from __future__ import annotations

import pytest

from orderflow_ai.core.config import PrimitivesConfig
from orderflow_ai.core.models import Side
from orderflow_ai.core.timeframes import Timeframe
from orderflow_ai.primitives import PrimitiveEngine
from orderflow_ai.primitives.direction import (
    compute_cvd,
    compute_delta,
    compute_delta_divergence,
)
from orderflow_ai.primitives.pressure import compute_imbalance, compute_orderbook_depth
from orderflow_ai.primitives.quality import compute_absorption, compute_volume_intensity
from orderflow_ai.primitives.statistics import (
    clamp,
    linear_slope,
    mean,
    percentile_rank,
    safe_div,
    stdev,
)
from orderflow_ai.primitives.structure import (
    compute_footprint,
    compute_value_migration,
    compute_volume_profile,
    poc_series,
    point_of_control,
)
from tests.factories import DEFAULT_START, make_bar, make_book, make_trades


def history(count: int = 5, base: float = 100.0, step: float = 0.5):
    """Build a simple rising history of bars."""
    return tuple(
        make_bar(
            open_ts=DEFAULT_START + index * 60,
            levels={base + index * step: (1.0, 1.0), base + index * step + 0.25: (1.0, 1.0)},
        )
        for index in range(count)
    )


class TestStatistics:
    def test_mean_of_empty_is_zero(self) -> None:
        assert mean([]) == 0.0

    def test_stdev_needs_two_points(self) -> None:
        assert stdev([5.0]) == 0.0

    def test_stdev_of_a_flat_series_is_zero(self) -> None:
        assert stdev([2.0, 2.0, 2.0]) == pytest.approx(0.0)

    def test_safe_div_guards_zero(self) -> None:
        assert safe_div(1.0, 0.0) == 0.0
        assert safe_div(1.0, 0.0, default=9.0) == 9.0

    def test_clamp_bounds_the_value(self) -> None:
        assert clamp(5.0) == 1.0
        assert clamp(-5.0) == 0.0
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_linear_slope_of_a_rising_series(self) -> None:
        assert linear_slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_linear_slope_of_short_series(self) -> None:
        assert linear_slope([]) == 0.0
        assert linear_slope([4.0]) == 0.0

    def test_percentile_rank_of_empty_sample(self) -> None:
        assert percentile_rank(1.0, []) == 0.0

    def test_percentile_rank_counts_below(self) -> None:
        assert percentile_rank(2.0, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.5)


class TestStructurePrimitives:
    def test_footprint_mirrors_the_bar(self) -> None:
        bar = make_bar(levels={100.0: (2.0, 3.0), 100.5: (1.0, 0.5)})
        footprint = compute_footprint(bar)

        assert footprint.total_volume == pytest.approx(6.5)
        assert footprint.buy_volume == pytest.approx(3.5)
        assert footprint.sell_volume == pytest.approx(3.0)
        assert footprint.delta == pytest.approx(0.5)
        assert len(footprint.levels) == 2
        assert footprint.levels[0].price == pytest.approx(100.0)

    def test_footprint_can_be_truncated_to_the_busiest_levels(self) -> None:
        bar = make_bar(
            levels={100.0: (1.0, 1.0), 100.5: (10.0, 10.0), 101.0: (2.0, 2.0)},
        )
        footprint = compute_footprint(bar, max_levels=1)
        assert len(footprint.levels) == 1
        assert footprint.levels[0].price == pytest.approx(100.5)

    def test_volume_profile_finds_the_point_of_control(self) -> None:
        bar = make_bar(levels={100.0: (1.0, 1.0), 100.5: (9.0, 9.0), 101.0: (1.0, 1.0)})
        profile = compute_volume_profile([bar])
        assert profile.poc == pytest.approx(100.5)
        assert profile.val <= profile.poc <= profile.vah

    def test_volume_profile_of_an_empty_history(self) -> None:
        profile = compute_volume_profile([])
        assert profile.total_volume == 0.0
        assert profile.poc == 0.0

    def test_volume_profile_identifies_high_volume_nodes(self) -> None:
        bars = [
            make_bar(open_ts=DEFAULT_START, levels={100.0: (5.0, 5.0), 100.5: (0.1, 0.1)}),
            make_bar(open_ts=DEFAULT_START + 60, levels={100.5: (1.0, 1.0)}),
        ]
        profile = compute_volume_profile(bars)
        assert 100.0 in profile.hvn

    def test_point_of_control_ignores_empty_bars(self) -> None:
        assert point_of_control(make_bar()) is None
        assert poc_series([make_bar()]) == []

    def test_value_migration_reports_upward_moves(self) -> None:
        bar = make_bar(levels={101.0: (5.0, 5.0)})
        profile = compute_volume_profile([bar])
        migration = compute_value_migration(profile, previous_pocs=[100.0], tick_size=0.5)
        assert migration.poc_direction == "up"
        assert migration.ticks == pytest.approx(2.0)
        assert migration.migration_strength > 0

    def test_value_migration_is_flat_without_history(self) -> None:
        profile = compute_volume_profile([make_bar(levels={100.0: (1.0, 1.0)})])
        migration = compute_value_migration(profile, previous_pocs=[], tick_size=0.5)
        assert migration.poc_direction == "flat"
        assert migration.migration_strength == 0.0


class TestDirectionPrimitives:
    def test_delta_splits_by_aggressor(self) -> None:
        bar = make_bar(levels={100.0: (2.0, 6.0)})
        delta = compute_delta(bar, 0.15)
        assert delta.value == pytest.approx(4.0)
        assert delta.direction == "bullish"
        assert delta.ratio == pytest.approx(0.5)

    def test_delta_is_neutral_inside_the_threshold(self) -> None:
        bar = make_bar(levels={100.0: (4.0, 6.0)})
        assert compute_delta(bar, 0.5).direction == "neutral"

    def test_delta_handles_an_empty_bar(self) -> None:
        delta = compute_delta(make_bar(), 0.15)
        assert delta.value == 0.0 and delta.ratio == 0.0

    def test_cvd_accumulates_over_the_lookback(self) -> None:
        bars = history(4)
        cvd = compute_cvd(bars[-1], bars[:-1], lookback=10)
        assert cvd.bars == 4
        assert cvd.value == pytest.approx(0.0)
        assert cvd.momentum == "neutral"

    def test_cvd_reports_accelerating_buying(self) -> None:
        quiet = [
            make_bar(open_ts=DEFAULT_START + index * 60, levels={100.0: (1.0, 2.0)})
            for index in range(4)
        ]
        burst = make_bar(open_ts=DEFAULT_START + 240, levels={100.0: (1.0, 20.0)})
        cvd = compute_cvd(burst, quiet, lookback=10)
        assert cvd.value > 0
        assert cvd.momentum == "accelerating"

    def test_cvd_reports_fading_selling(self) -> None:
        calm = [
            make_bar(open_ts=DEFAULT_START + index * 60, levels={100.0: (1.0, 2.0)})
            for index in range(4)
        ]
        selling = make_bar(open_ts=DEFAULT_START + 240, levels={100.0: (30.0, 1.0)})
        cvd = compute_cvd(selling, calm, lookback=10)
        assert cvd.value < 0
        assert cvd.momentum == "fading"

    def test_delta_divergence_needs_enough_bars(self) -> None:
        bars = history(2)
        divergence = compute_delta_divergence(bars[-1], bars[:-1], min_bars=5)
        assert not divergence.detected
        assert divergence.bars_observed == 2

    def test_delta_divergence_detects_price_up_delta_down(self) -> None:
        bars = [
            make_bar(open_ts=DEFAULT_START + index * 60, levels={100.0 + index: (5.0, 1.0)})
            for index in range(5)
        ]
        divergence = compute_delta_divergence(bars[-1], bars[:-1], min_bars=3)
        assert divergence.detected
        assert divergence.type == "bearish"
        assert 0 < divergence.confidence <= 100


class TestPressurePrimitives:
    def test_imbalance_reports_levels_above_the_threshold(self) -> None:
        bar = make_bar(levels={100.0: (1.0, 4.0), 100.5: (2.0, 2.0)})
        imbalance = compute_imbalance(bar, threshold=2.0)
        assert imbalance.detected
        assert len(imbalance.levels) == 1
        assert imbalance.levels[0].side is Side.BUY
        assert imbalance.levels[0].ratio == pytest.approx(4.0)

    def test_imbalance_always_reports_the_strongest_level(self) -> None:
        bar = make_bar(levels={100.0: (1.0, 1.5)})
        imbalance = compute_imbalance(bar, threshold=5.0)
        assert not imbalance.detected
        assert imbalance.strongest is not None
        assert imbalance.strongest.ratio == pytest.approx(1.5)

    def test_imbalance_ignores_one_sided_levels(self) -> None:
        bar = make_bar(levels={100.0: (5.0, 0.0)})
        assert compute_imbalance(bar).strongest is None

    def test_orderbook_depth_reports_direction(self) -> None:
        depth = compute_orderbook_depth(make_book(bid_size=20.0, ask_size=10.0), levels=5)
        assert depth.imbalance_direction == "bid"
        assert depth.imbalance_ratio == pytest.approx(2.0)
        assert depth.total_depth == pytest.approx(150.0)

    def test_orderbook_depth_is_balanced_by_default(self) -> None:
        depth = compute_orderbook_depth(make_book(), levels=5)
        assert depth.imbalance_direction == "balanced"

    def test_orderbook_depth_finds_walls(self) -> None:
        from orderflow_ai.core.models import BookSnapshot, DepthLevel

        book = BookSnapshot(
            bids=[DepthLevel(99.5, 10.0), DepthLevel(99.0, 200.0), DepthLevel(98.5, 10.0)],
            asks=[DepthLevel(100.5, 10.0), DepthLevel(101.0, 10.0)],
        )
        depth = compute_orderbook_depth(book, levels=5)
        assert depth.walls
        assert depth.walls[0].price == pytest.approx(99.0)
        assert depth.walls[0].side is Side.BUY

    def test_orderbook_depth_of_an_empty_book(self) -> None:
        from orderflow_ai.core.models import BookSnapshot

        depth = compute_orderbook_depth(BookSnapshot.empty(), levels=5)
        assert depth.total_depth == 0.0
        assert depth.spread_bps is None


class TestQualityPrimitives:
    def test_volume_intensity_without_history_is_normal(self) -> None:
        bar = make_bar(levels={100.0: (1.0, 1.0)})
        intensity = compute_volume_intensity(bar, [])
        assert intensity.ratio == pytest.approx(1.0)
        assert intensity.intensity == "normal"

    def test_volume_intensity_detects_heavy_volume(self) -> None:
        past = history(5)
        heavy = make_bar(levels={100.0: (20.0, 20.0)})
        intensity = compute_volume_intensity(heavy, past)
        assert intensity.ratio > 2.5
        assert intensity.intensity == "extreme"
        assert intensity.percentile == pytest.approx(100.0)

    def test_volume_intensity_detects_thin_volume(self) -> None:
        past = history(5)
        thin = make_bar(levels={100.0: (0.05, 0.05)})
        assert compute_volume_intensity(thin, past).intensity == "low"

    def test_absorption_scores_heavy_but_compressed_bars(self) -> None:
        past = history(5)
        absorbed = make_bar(levels={100.0: (1.0, 40.0)}, trades_per_level=6)
        score = compute_absorption(absorbed, past, min_confidence=50.0)
        assert score.detected
        assert score.confidence > 50
        assert score.side is Side.BUY
        assert score.price == pytest.approx(100.0)

    def test_absorption_ignores_normal_bars(self) -> None:
        past = history(5)
        normal = make_bar(levels={100.0: (1.0, 1.0)})
        score = compute_absorption(normal, past, min_confidence=60.0)
        assert not score.detected

    def test_absorption_of_an_empty_bar(self) -> None:
        score = compute_absorption(make_bar(), history(3))
        assert score.confidence == 0.0
        assert score.side is None


class TestPrimitiveEngine:
    def test_computes_all_ten_primitives(self) -> None:
        bars = history(6)
        engine = PrimitiveEngine(PrimitivesConfig(), tick_size=0.25, depth_levels=10)
        primitives = engine.compute(bars[-1], bars[:-1], make_book())

        assert primitives.symbol == "BTCUSDT"
        assert primitives.timeframe is Timeframe.M1
        assert primitives.structure.footprint.total_volume > 0
        assert primitives.structure.volume_profile.poc > 0
        assert primitives.direction.delta.value == pytest.approx(0.0)
        assert primitives.direction.cvd.bars == 6
        assert isinstance(primitives.pressure.imbalance.threshold, float)
        assert primitives.pressure.orderbook.imbalance_ratio > 0
        assert primitives.quality.volume_intensity.intensity
        assert primitives.quality.absorption.confidence >= 0

    def test_bias_follows_delta_and_cvd(self) -> None:
        bars = history(6)
        engine = PrimitiveEngine()
        bullish = engine.compute(
            make_bar(open_ts=bars[-1].open_time.timestamp() + 60, levels={100.0: (1.0, 9.0)}),
            bars,
            make_book(),
        )
        assert bullish.bias == "bullish"

        bearish = engine.compute(
            make_bar(open_ts=bars[-1].open_time.timestamp() + 120, levels={100.0: (9.0, 1.0)}),
            bars,
            make_book(),
        )
        assert bearish.bias == "bearish"

    def test_is_deterministic(self) -> None:
        bars = history(4)
        engine = PrimitiveEngine()
        first = engine.compute(bars[-1], bars[:-1], make_book())
        second = engine.compute(bars[-1], bars[:-1], make_book())
        assert first == second

    def test_handles_a_first_bar_with_no_history(self) -> None:
        engine = PrimitiveEngine()
        primitives = engine.compute(make_bar(levels={100.0: (1.0, 2.0)}))
        assert primitives.direction.cvd.bars == 1
        assert primitives.quality.volume_intensity.ratio == pytest.approx(1.0)

    def test_accepts_a_history_of_trades_built_bars(self) -> None:
        from orderflow_ai.ingestion import CandleBuilder

        builder = CandleBuilder(Timeframe.M1, price_decimals=2)
        for trade in make_trades([(100.0 + index * 0.5, 2.0, Side.BUY) for index in range(4)]):
            builder.push(trade)
        bar = builder.flush()
        assert bar is not None
        primitives = PrimitiveEngine().compute(bar)
        assert primitives.price == pytest.approx(101.5)