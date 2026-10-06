"""Layer 3: pattern detection, ranking and the no-pattern fallback."""

from __future__ import annotations

import pytest

from orderflow_ai.core.config import PatternsConfig, PrimitivesConfig
from orderflow_ai.patterns import FallbackContext, FallbackScorer, PatternContext, PatternEngine
from orderflow_ai.patterns.base import PatternDetector
from orderflow_ai.patterns.models import PatternMatch, classify_tier
from orderflow_ai.primitives import PrimitiveEngine
from tests.factories import DEFAULT_START, make_bar, make_book

#: Prices used throughout the module, one tick apart.
BASE = 100.0
TICK = 0.5


def quiet_bar(index: int, price: float = BASE) -> object:
    """A neutral two level bar with a real range, used as history."""
    return make_bar(
        open_ts=DEFAULT_START + index * 60,
        levels={price - TICK: (2.0, 2.0), price: (2.0, 2.0)},
    )


def event_bar(
    index: int,
    levels: dict[float, tuple[float, float]],
    trades: int = 1,
    close: float | None = None,
):
    """A bar built for one specific pattern."""
    return make_bar(
        open_ts=DEFAULT_START + index * 60,
        levels=levels,
        trades_per_level=trades,
        close=close,
    )


def primitives_for(bars, book=None):
    """Compute primitives for a sequence of bars."""
    return PrimitiveEngine(PrimitivesConfig()).compute(
        bars[-1], tuple(bars[:-1]), book or make_book()
    )


def context_for(bars, supports=(), resistances=(), tick_size: float = TICK) -> PatternContext:
    """Build a pattern context for a sequence of bars."""
    return PatternContext(
        primitives=primitives_for(bars),
        bars=tuple(bars),
        supports=supports,
        resistances=resistances,
        tick_size=tick_size,
        price_decimals=2,
    )


class TestConfidenceTiers:
    @pytest.mark.parametrize(
        ("score", "tier"),
        [(100, "high"), (75, "high"), (74, "medium"), (50, "medium"), (49, "low"), (0, "none")],
    )
    def test_tiers_match_the_specification(self, score: int, tier: str) -> None:
        assert classify_tier(score) == tier


class TestFallbackScorer:
    def test_scores_at_most_one_hundred(self) -> None:
        bars = [*quiet_history(5), event_bar(5, {BASE: (1.0, 9.0)})]
        score = FallbackScorer().score(
            primitives_for(bars),
            FallbackContext(mtf_alignment=1.0, level_proximity=1.0, volatility_regime="normal"),
        )
        assert 0 <= score.score <= 100
        assert score.tier in {"high", "medium", "low", "none"}

    def test_weights_sum_to_one_hundred(self) -> None:
        assert sum(FallbackScorer().weights.values()) == pytest.approx(100.0)

    def test_better_context_scores_higher(self) -> None:
        bars = quiet_history(5)
        primitives = primitives_for(bars)
        weak = FallbackScorer().score(primitives, FallbackContext())
        strong = FallbackScorer().score(
            primitives,
            FallbackContext(mtf_alignment=1.0, level_proximity=1.0, volatility_regime="normal"),
        )
        assert strong.score > weak.score

    def test_breaks_the_score_into_components(self) -> None:
        score = FallbackScorer().score(primitives_for(quiet_history(3)))
        assert [component.name for component in score.components] == [
            "delta_strength",
            "orderbook_imbalance",
            "mtf_alignment",
            "volume_intensity",
            "level_proximity",
            "volatility_regime",
        ]
        assert all(component.max_points > 0 for component in score.components)

    def test_survives_missing_context(self) -> None:
        assert FallbackScorer().score(primitives_for(quiet_history(3))).score >= 0


def quiet_history(count: int, price: float = BASE) -> list:
    """``count`` neutral bars."""
    return [quiet_bar(index, price) for index in range(count)]


def _fixed_detector(confidence: float) -> PatternDetector:
    """A detector that always reports the same confidence."""

    class Fixed(PatternDetector):
        name = "fixed"
        label = "Fixed"

        def detect(self, context: PatternContext) -> PatternMatch:
            return PatternMatch(
                name=self.name,
                label=self.label,
                confidence=confidence,
                direction="bullish",
                price=BASE,
            )

    return Fixed()


class TestAbsorptionDetector:
    def test_fires_on_heavy_but_compressed_volume(self) -> None:
        bars = [*quiet_history(5), event_bar(5, {BASE: (1.0, 40.0)}, trades=6)]
        result = PatternEngine().detect(context_for(bars))

        assert "absorption" in result.names
        absorption = next(match for match in result.patterns if match.name == "absorption")
        assert absorption.direction == "bullish"
        assert absorption.price == pytest.approx(BASE)
        assert absorption.evidence


class TestIcebergDetector:
    def test_fires_on_many_small_clips_at_one_level(self) -> None:
        # The suspect level takes most of the volume in many tiny clips, while
        # the neighbouring level trades in large prints.
        iceberg = event_bar(4, {BASE - TICK: (4.0, 4.0), BASE: (10.0, 10.0)})
        iceberg.levels[BASE].bid_trades = 30
        iceberg.levels[BASE].ask_trades = 30
        bars = [*quiet_history(4), iceberg]

        assert "iceberg" in PatternEngine().detect(context_for(bars)).names

    def test_stays_quiet_without_a_clip_size_contrast(self) -> None:
        uniform = event_bar(4, {BASE - TICK: (3.0, 3.0), BASE: (3.0, 3.0)}, trades=10)
        bars = [*quiet_history(4), uniform]

        assert "iceberg" not in PatternEngine().detect(context_for(bars)).names


class TestTrappedTradersDetector:
    def _bars(self) -> list:
        breakout = event_bar(3, {BASE: (1.0, 1.0), BASE + TICK: (1.0, 12.0), BASE + 2 * TICK: (1.0, 12.0)})
        rejection = event_bar(4, {BASE - TICK: (1.0, 1.0), BASE: (14.0, 1.0)})
        return [*quiet_history(3), breakout, rejection]

    def test_detects_bullish_buyers_trapped_above_resistance(self) -> None:
        context = context_for(self._bars(), resistances=(BASE + 0.25,))
        result = PatternEngine().detect(context)

        assert "trapped_traders" in result.names
        trapped = next(match for match in result.patterns if match.name == "trapped_traders")
        assert trapped.direction == "bearish"
        assert trapped.price == pytest.approx(BASE + 0.25)
        assert trapped.bars_ago == 1

    def test_ignores_breakouts_without_aggressive_delta(self) -> None:
        breakout = event_bar(3, {BASE: (1.0, 1.0), BASE + 2 * TICK: (1.0, 1.0)})
        rejection = event_bar(4, {BASE - TICK: (10.0, 1.0), BASE: (5.0, 1.0)})
        context = context_for([*quiet_history(3), breakout, rejection], resistances=(BASE + 0.25,))

        assert "trapped_traders" not in PatternEngine().detect(context).names


class TestLiquiditySweepDetector:
    def _history(self) -> list:
        return quiet_history(3, price=BASE)

    def test_detects_a_thin_overshoot_below_resistance(self) -> None:
        sweep = event_bar(3, {BASE + TICK / 2: (6.0, 5.0), BASE - TICK / 2: (4.0, 4.0)}, close=BASE - TICK / 2)
        context = context_for([*self._history(), sweep], resistances=(BASE,))
        result = PatternEngine().detect(context)

        assert "liquidity_sweep" in result.names
        sweep_match = next(match for match in result.patterns if match.name == "liquidity_sweep")
        assert sweep_match.direction == "bearish"
        assert sweep_match.price == pytest.approx(BASE)

    def test_detects_a_thin_overshoot_above_support(self) -> None:
        sweep = event_bar(3, {BASE - TICK / 2: (5.0, 6.0), BASE + TICK / 2: (4.0, 4.0)}, close=BASE + TICK / 2)
        context = context_for([*self._history(), sweep], supports=(BASE,))
        result = PatternEngine().detect(context)

        assert "liquidity_sweep" in result.names
        sweep_match = next(match for match in result.patterns if match.name == "liquidity_sweep")
        assert sweep_match.direction == "bullish"

    def test_requires_a_weak_delta(self) -> None:
        breakout = event_bar(3, {BASE + TICK: (1.0, 20.0), BASE: (1.0, 1.0)}, close=BASE)
        context = context_for([*self._history(), breakout], resistances=(BASE,))

        assert "liquidity_sweep" not in PatternEngine().detect(context).names


class TestDeltaDivergenceDetector:
    def test_reports_a_bearish_divergence(self) -> None:
        bars = [
            event_bar(index, {BASE + index * TICK: (6.0, 1.0), BASE + index * TICK + 0.25: (1.0, 1.0)})
            for index in range(5)
        ]
        result = PatternEngine().detect(context_for(bars))

        assert "delta_divergence" in result.names
        divergence = next(match for match in result.patterns if match.name == "delta_divergence")
        assert divergence.direction == "bearish"


class TestExhaustionDetector:
    def test_reports_a_fading_uptrend(self) -> None:
        trend = [
            event_bar(index, {BASE + index * TICK: (2.0, 8.0), BASE + index * TICK - TICK: (2.0, 2.0)})
            for index in range(4)
        ]
        exhausted = event_bar(
            4, {BASE + 2 * TICK: (2.0, 3.0), BASE + 1.5 * TICK: (1.0, 1.0)}, close=BASE + 2 * TICK
        )
        result = PatternEngine().detect(context_for([*trend, exhausted]))

        assert "exhaustion" in result.names
        exhaustion = next(match for match in result.patterns if match.name == "exhaustion")
        assert exhaustion.direction == "bearish"
        assert exhaustion.metrics["volume_decay"] > 0.15


class TestUnfinishedAuctionDetector:
    def test_detects_unlifted_offers_at_the_high(self) -> None:
        unfinished = event_bar(4, {BASE: (2.0, 2.0), BASE + TICK: (10.0, 0.0)})
        result = PatternEngine().detect(context_for([*quiet_history(4), unfinished]))

        assert "unfinished_auction" in result.names
        auction = next(match for match in result.patterns if match.name == "unfinished_auction")
        assert auction.direction == "bearish"
        assert auction.price == pytest.approx(BASE + TICK)

    def test_detects_untouched_bids_at_the_low(self) -> None:
        unfinished = event_bar(4, {BASE - TICK: (0.0, 10.0), BASE: (2.0, 2.0)})
        result = PatternEngine().detect(context_for([*quiet_history(4), unfinished]))

        assert "unfinished_auction" in result.names
        auction = next(match for match in result.patterns if match.name == "unfinished_auction")
        assert auction.direction == "bullish"


class TestPatternEngine:
    def test_returns_the_fallback_when_nothing_matches(self) -> None:
        result = PatternEngine(PatternsConfig(min_confidence=99.0)).detect(
            context_for(quiet_history(4))
        )
        assert not result.detected
        assert result.fallback is not None
        assert result.confidence == result.fallback.score
        assert result.direction == "neutral"

    def test_filters_below_min_confidence(self) -> None:
        result = PatternEngine(
            PatternsConfig(min_confidence=50.0), detectors=[_fixed_detector(40.0)]
        ).detect(context_for(quiet_history(3)))
        assert not result.detected

    def test_accepts_above_min_confidence(self) -> None:
        result = PatternEngine(
            PatternsConfig(min_confidence=30.0), detectors=[_fixed_detector(40.0)]
        ).detect(context_for(quiet_history(3)))
        assert result.detected
        assert result.confidence == pytest.approx(40.0)

    def test_caps_the_number_of_patterns(self) -> None:
        bars = [
            *quiet_history(4),
            event_bar(4, {BASE - TICK: (1.0, 40.0), BASE: (40.0, 1.0)}, trades=6),
        ]
        engine = PatternEngine(PatternsConfig(min_confidence=0.0, max_patterns=1))
        assert len(engine.detect(context_for(bars)).patterns) <= 1

    def test_drops_patterns_older_than_the_window(self) -> None:
        breakout = event_bar(3, {BASE: (1.0, 1.0), BASE + 2 * TICK: (1.0, 12.0)})
        rejection = event_bar(4, {BASE - 2 * TICK: (20.0, 1.0), BASE - TICK: (5.0, 1.0)})
        context = context_for([*quiet_history(3), breakout, rejection], resistances=(BASE + 0.25,))

        strict = PatternEngine(PatternsConfig(min_confidence=0.0, max_age_bars=0))
        loose = PatternEngine(PatternsConfig(min_confidence=0.0, max_age_bars=3))
        assert "trapped_traders" not in strict.detect(context).names
        assert "trapped_traders" in loose.detect(context).names

    def test_ranks_by_confidence(self) -> None:
        bars = [
            *quiet_history(4),
            event_bar(4, {BASE - TICK: (1.0, 40.0), BASE: (40.0, 1.0)}, trades=6),
        ]
        engine = PatternEngine(PatternsConfig(min_confidence=0.0))
        confidences = [match.confidence for match in engine.detect(context_for(bars)).patterns]
        assert confidences == sorted(confidences, reverse=True)

    def test_removes_duplicate_patterns(self) -> None:
        bars = [
            *quiet_history(4),
            event_bar(4, {BASE - TICK: (1.0, 40.0), BASE: (40.0, 1.0)}, trades=6),
        ]
        result = PatternEngine(PatternsConfig(min_confidence=0.0)).detect(context_for(bars))
        keys = [(match.name, match.price) for match in result.patterns]
        assert len(keys) == len(set(keys))

    def test_survives_a_broken_detector(self) -> None:
        class Exploding(PatternDetector):
            name = "exploding"
            label = "Exploding"

            def detect(self, context: PatternContext) -> PatternMatch | None:
                raise RuntimeError("boom")

        result = PatternEngine(detectors=[Exploding()]).detect(context_for(quiet_history(3)))
        assert not result.detected
        assert result.fallback is not None

    def test_rejects_more_detectors_than_the_specification(self) -> None:
        class Extra(PatternDetector):
            name = "extra"

            def detect(self, context: PatternContext) -> PatternMatch | None:
                return None

        with pytest.raises(ValueError, match="detectors"):
            PatternEngine(detectors=[Extra() for _ in range(8)])

    def test_exposes_detector_names(self) -> None:
        assert PatternEngine().detector_names == (
            "absorption",
            "trapped_traders",
            "liquidity_sweep",
            "delta_divergence",
            "exhaustion",
            "iceberg",
            "unfinished_auction",
        )

    def test_result_helpers(self) -> None:
        bars = [*quiet_history(4), event_bar(4, {BASE: (1.0, 40.0)}, trades=6)]
        result = PatternEngine(PatternsConfig(min_confidence=0.0)).detect(context_for(bars))
        assert result.best is not None
        assert result.best.name in result.names
        assert result.filter_by_confidence(result.best.confidence)