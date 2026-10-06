"""Layer 4: market structure, key levels, sessions, volatility and alignment."""

from __future__ import annotations

import datetime as dt

import pytest

from orderflow_ai.context import (
    ContextBuilder,
    Session,
    build_key_levels,
    build_multi_timeframe,
    build_session_context,
    build_structure,
    build_volatility,
    detect_swings,
    round_numbers,
    session_of,
)
from orderflow_ai.context.models import KeyLevels
from orderflow_ai.core.config import PrimitivesConfig
from orderflow_ai.core.timeframes import Timeframe
from orderflow_ai.primitives import PrimitiveEngine
from orderflow_ai.primitives.models import VolumeProfile
from tests.factories import make_bar, make_book, make_candle, zigzag_candles


def primitives_for(bars):
    """Compute primitives for a sequence of footprint bars."""
    return PrimitiveEngine(PrimitivesConfig()).compute(
        bars[-1], tuple(bars[:-1]), make_book()
    )


class TestSwings:
    def test_finds_confirmed_pivots_on_a_zigzag(self) -> None:
        swings = detect_swings(zigzag_candles(20))
        assert [(swing.index % 8, swing.is_high) for swing in swings] == [
            (2, True),
            (6, False),
            (2, True),
            (6, False),
        ]

    def test_needs_bars_on_both_sides_of_a_pivot(self) -> None:
        assert detect_swings(zigzag_candles(4)) == ()

    def test_rejects_equal_neighbours(self) -> None:
        flat = [make_candle(index=i, open_=100.0, high=101.0, low=99.0, close=100.0) for i in range(9)]
        assert detect_swings(flat) == ()

    def test_returns_unknown_without_swings(self) -> None:
        structure = build_structure(zigzag_candles(4))
        assert structure.label == "unknown"
        assert structure.bos == "none"


class TestMarketStructure:
    def test_labels_a_rising_zigzag_as_an_uptrend(self) -> None:
        structure = build_structure(zigzag_candles(20))
        assert structure.label == "uptrend"
        assert structure.is_bullish
        assert [round(level, 3) for level in structure.swing_highs] == [102.4, 103.2]
        assert [round(level, 3) for level in structure.swing_lows] == [98.4, 99.2]

    def test_reports_a_bullish_break_of_structure(self) -> None:
        assert build_structure(zigzag_candles(20)).bos == "bullish"

    def test_reports_a_bearish_break_of_structure(self) -> None:
        assert build_structure(zigzag_candles(count=22, trend=0.3)).bos == "bearish"

    def test_flags_a_change_of_character(self) -> None:
        structure = build_structure(zigzag_candles(24, dampen_after=12))
        assert structure.choch
        assert structure.label == "range"

    def test_labels_a_flat_series_as_a_range(self) -> None:
        structure = build_structure(zigzag_candles(20, trend=0.0))
        assert structure.label in {"range", "unknown"}


class TestKeyLevels:
    def test_splits_levels_around_price(self) -> None:
        profile = VolumeProfile(poc=100.0, vah=101.0, val=99.0, hvn=(100.5,), lvn=(98.5,))
        levels = build_key_levels(price=100.0, profile=profile, max_levels=3)
        assert all(level < 100.0 for level in levels.support)
        assert all(level > 100.0 for level in levels.resistance)
        assert 100.5 in levels.resistance
        assert 98.5 in levels.support

    def test_keeps_the_nearest_level_first(self) -> None:
        profile = VolumeProfile(poc=100.0, vah=101.0, val=99.0, hvn=(102.0, 103.0), lvn=(95.0, 96.0))
        levels = build_key_levels(price=100.0, profile=profile, max_levels=2)
        assert levels.resistance == (101.0, 102.0)
        assert levels.support == (99.0, 96.0)

    def test_merges_levels_within_tolerance(self) -> None:
        profile = VolumeProfile(poc=100.0, vah=100.4, val=99.6)
        merged = build_key_levels(price=100.0, profile=profile, tolerance=1.0)
        assert merged.resistance == (100.4,)
        assert merged.support == (99.6,)

    def test_uses_swing_points_when_available(self) -> None:
        from orderflow_ai.context.structure import Swing

        swings = (
            Swing(price=103.0, index=1, is_high=True),
            Swing(price=97.0, index=2, is_high=False),
        )
        levels = build_key_levels(
            price=100.0, profile=VolumeProfile(poc=100.0, vah=101.0, val=99.0), swings=swings
        )
        assert 103.0 in levels.resistance
        assert 97.0 in levels.support

    def test_nearest_helpers(self) -> None:
        levels = KeyLevels(support=(95.0, 99.0), resistance=(101.0, 105.0))
        assert levels.nearest_support(100.0) == 99.0
        assert levels.nearest_resistance(100.0) == 101.0
        assert levels.nearest_support(90.0) is None
        assert levels.nearest_resistance(110.0) is None

    def test_round_numbers_bracket_price(self) -> None:
        numbers = round_numbers(101.37, count=2)
        assert any(level > 101.37 for level in numbers)
        assert any(level < 101.37 for level in numbers)


class TestSessions:
    @pytest.mark.parametrize(
        ("hour", "session"),
        [(2, Session.ASIA), (8, Session.LONDON), (22, Session.OFF_HOURS)],
    )
    def test_session_of_an_hour(self, hour: int, session: Session) -> None:
        assert session_of(dt.datetime(2024, 1, 1, hour, tzinfo=dt.UTC)) is session

    def test_london_stays_primary_during_the_overlap(self) -> None:
        moment = dt.datetime(2024, 1, 1, 13, tzinfo=dt.UTC)
        context = build_session_context(moment)
        assert context.primary == "london"
        assert set(context.active) == {"london", "new_york"}

    def test_new_york_takes_over_after_london_closes(self) -> None:
        moment = dt.datetime(2024, 1, 1, 17, tzinfo=dt.UTC)
        context = build_session_context(moment)
        assert context.primary == "new_york"
        assert context.active == ("new_york",)

    def test_flags_the_london_open(self) -> None:
        context = build_session_context(dt.datetime(2024, 1, 1, 7, 10, tzinfo=dt.UTC))
        assert context.is_london_open
        assert context.minutes_into_session == pytest.approx(10.0)

    def test_flags_the_new_york_open(self) -> None:
        moment = dt.datetime(2024, 1, 1, 12, 15, tzinfo=dt.UTC)
        context = build_session_context(moment)
        assert context.is_new_york_open
        assert not context.is_london_open


class TestVolatility:
    def test_steady_range_is_normal(self) -> None:
        candles = [make_candle(index=i, high=101.0, low=99.0) for i in range(12)]
        assert build_volatility(candles).regime == "normal"

    def test_expanding_range_is_high(self) -> None:
        candles = [make_candle(index=i, high=101.0, low=99.0) for i in range(12)]
        candles.append(make_candle(index=12, high=110.0, low=90.0))
        assert build_volatility(candles).regime in {"high", "extreme"}

    def test_contracting_range_is_low(self) -> None:
        candles = [make_candle(index=i, high=101.0, low=99.0) for i in range(12)]
        candles.append(make_candle(index=12, high=100.1, low=99.9))
        assert build_volatility(candles).regime == "low"

    def test_reports_the_average_range(self) -> None:
        candles = [make_candle(index=i, high=102.0, low=98.0) for i in range(5)]
        assert build_volatility(candles).average_range == pytest.approx(4.0)

    def test_empty_series_is_normal(self) -> None:
        assert build_volatility([]).regime == "normal"


class TestMultiTimeframe:
    def test_aligned_when_all_timeframes_agree(self) -> None:
        candles = {
            Timeframe.M5: zigzag_candles(10),
            Timeframe.M15: zigzag_candles(10),
        }
        result = build_multi_timeframe(candles)
        assert result.direction == "aligned"
        assert result.score == pytest.approx(100.0)
        assert result.aligned

    def test_conflict_when_timeframes_disagree(self) -> None:
        candles = {
            Timeframe.M5: zigzag_candles(10, trend=0.5),
            Timeframe.M15: zigzag_candles(10, trend=-0.5),
        }
        result = build_multi_timeframe(candles)
        assert result.direction == "conflict"
        assert result.score == pytest.approx(50.0)

    def test_ignores_timeframes_without_history(self) -> None:
        candles = {Timeframe.M5: [make_candle()], Timeframe.M15: []}
        assert build_multi_timeframe(candles).direction == "mixed"

    def test_flat_timeframes_are_neutral(self) -> None:
        candles = {Timeframe.M5: [make_candle(index=i, high=100.0, low=100.0) for i in range(3)]}
        assert build_multi_timeframe(candles).direction == "mixed"

    def test_reports_each_timeframe_state(self) -> None:
        candles = {Timeframe.M5: zigzag_candles(6), Timeframe.M15: zigzag_candles(6)}
        states = build_multi_timeframe(candles).states
        assert [state.timeframe for state in states] == [Timeframe.M5, Timeframe.M15]
        assert all(state.bars == 6 for state in states)


class TestContextBuilder:
    def _bars(self, count: int = 8) -> list:
        return [
            make_bar(
                open_ts=1_700_000_000.0 + index * 60,
                levels={
                    100.0 + index * 0.5: (2.0, 3.0),
                    99.5 + index * 0.5: (1.0, 1.0),
                    100.5 + index * 0.5: (1.0, 1.0),
                },
            )
            for index in range(count)
        ]

    def test_builds_a_complete_context(self) -> None:
        bars = self._bars()
        context = ContextBuilder().build(primitives_for(bars), bars)

        assert context.symbol == "BTCUSDT"
        assert context.timeframe is Timeframe.M1
        assert context.price == pytest.approx(primitives_for(bars).price)
        assert context.levels.poc is not None
        assert context.session.primary
        assert 0.0 <= context.level_proximity <= 1.0
        assert context.volatility.regime

    def test_level_proximity_is_higher_on_a_level(self) -> None:
        bars = self._bars()
        context = ContextBuilder(max_levels=1).build(primitives_for(bars), bars)
        nearest = context.nearest_level_price
        assert nearest is not None
        expected = max(0.0, 1.0 - abs(context.price - nearest) / context.volatility.average_range)
        assert context.level_proximity == pytest.approx(min(expected, 1.0))

    def test_level_proximity_is_zero_without_levels(self) -> None:
        context = ContextBuilder().build(primitives_for(self._bars()), self._bars())
        assert 0.0 <= context.level_proximity <= 1.0

    def test_accepts_higher_timeframe_candles(self) -> None:
        bars = self._bars()
        higher = {Timeframe.M5: zigzag_candles(6)}
        context = ContextBuilder().build(primitives_for(bars), bars, higher)
        assert context.multi_timeframe.states
        assert context.multi_timeframe.score > 0