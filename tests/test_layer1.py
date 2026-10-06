"""Layer 1: ingestion, normalisation, buffering and timestamp alignment."""

from __future__ import annotations

import datetime as dt

import pytest

from orderflow_ai.core.config import OrderFlowConfig
from orderflow_ai.core.models import DepthLevel, DepthUpdate, Side, Trade
from orderflow_ai.core.timeframes import Timeframe
from orderflow_ai.ingestion import (
    BinanceNormalizer,
    CandleBuilder,
    IngestionHub,
    OrderBook,
    ReplayFeed,
    RingBuffer,
    SyntheticFeed,
    TimestampSync,
    build_trade_feed,
)
from orderflow_ai.ingestion.resampler import resample_bars
from tests.factories import DEFAULT_START, make_bar, make_config, make_trade, make_trades


class TestRingBuffer:
    def test_retains_items_in_order(self) -> None:
        buffer = RingBuffer[int](3)
        buffer.extend([1, 2, 3])
        assert buffer.to_list() == [1, 2, 3]

    def test_overwrites_oldest_when_full(self) -> None:
        buffer = RingBuffer[int](3)
        buffer.extend([1, 2, 3, 4, 5])
        assert buffer.to_list() == [3, 4, 5]
        assert len(buffer) == 3

    def test_latest_returns_newest_items_oldest_first(self) -> None:
        buffer = RingBuffer[int](4)
        buffer.extend(range(5))
        assert buffer.latest(2) == [3, 4]

    def test_latest_clamps_to_available_items(self) -> None:
        buffer = RingBuffer[int](4)
        buffer.append(1)
        assert buffer.latest(10) == [1]

    def test_rejects_non_positive_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            RingBuffer[int](0)

    def test_clear_empties_the_buffer(self) -> None:
        buffer = RingBuffer[int](2)
        buffer.extend([1, 2])
        buffer.clear()
        assert not buffer and len(buffer) == 0


class TestTimestampSync:
    def test_converts_milliseconds_to_seconds(self) -> None:
        sync = TimestampSync()
        assert sync.normalize("trades", 1_700_000_000_000) == pytest.approx(1_700_000_000.0)

    def test_clamps_backwards_regressions(self) -> None:
        sync = TimestampSync(max_regression_ms=100)
        sync.normalize("trades", 1_700_000_010.0)
        assert sync.normalize("trades", 1_700_000_000.0) == 1_700_000_010.0

    def test_accepts_small_jitter(self) -> None:
        sync = TimestampSync(max_regression_ms=250)
        sync.normalize("trades", 1_700_000_010.0)
        assert sync.normalize("trades", 1_700_000_009.9) == 1_700_000_009.9

    def test_reports_per_source_diagnostics(self) -> None:
        sync = TimestampSync()
        sync.normalize("trades", 1_700_000_010.0)
        sync.normalize("depth", 1_700_000_008.0)
        report = sync.report()
        assert report["trades"]["last_timestamp"] == 1_700_000_010.0
        assert report["depth"]["lag_seconds"] == pytest.approx(2.0)

    def test_synchronize_returns_newest_sample(self) -> None:
        sync = TimestampSync()
        newest = sync.synchronize({"trades": 1_700_000_005.0, "depth": 1_700_000_009.0})
        assert newest == 1_700_000_009.0


class TestBinanceNormalizer:
    normalizer = BinanceNormalizer("BTCUSDT")

    def test_agg_trade_buyer_is_aggressor(self) -> None:
        trade = self.normalizer.trade({"e": "aggTrade", "s": "BTCUSDT", "p": "100.5", "q": "2", "m": False, "T": 1_700_000_000_000})
        assert trade is not None
        assert trade.side is Side.BUY
        assert trade.price == pytest.approx(100.5)
        assert trade.signed_size == pytest.approx(2.0)

    def test_agg_trade_maker_is_sell_aggressor(self) -> None:
        trade = self.normalizer.trade({"p": "100.5", "q": "2", "m": True, "T": 1_700_000_000_000})
        assert trade is not None
        assert trade.side is Side.SELL

    def test_trade_without_price_is_skipped(self) -> None:
        assert self.normalizer.trade({"p": "0", "q": "2", "T": 1}) is None

    def test_partial_book_event_is_a_snapshot(self) -> None:
        update = self.normalizer.depth(
            {"e": "depthUpdate", "E": 1_700_000_000_000, "bids": [["99", "1"]], "asks": [["101", "2"]]}
        )
        assert update is not None and update.is_snapshot
        assert update.asks[0] == DepthLevel(101.0, 2.0)

    def test_diff_event_is_not_a_snapshot(self) -> None:
        update = self.normalizer.depth(
            {"e": "depthUpdate", "E": 1_700_000_000_000, "b": [["99", "1"]], "a": [["101", "2"]]}
        )
        assert update is not None and not update.is_snapshot
        assert update.bids[0].price == pytest.approx(99.0)

    def test_rest_kline_row_becomes_candle(self) -> None:
        row = [1_700_000_000_000, "100", "105", "99", "104", "12.5", 1_700_000_059_999, "0", 42, "0", "0", "0"]
        candle = self.normalizer.candle(row, None)
        assert candle.open == 100.0 and candle.close == 104.0
        assert candle.high == 105.0 and candle.low == 99.0
        assert candle.trade_count == 42
        assert candle.open_time.tzinfo is dt.UTC

    def test_kline_event_becomes_candle(self) -> None:
        event = {"e": "kline", "k": {"t": 1_700_000_000_000, "T": 1_700_000_059_999, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "n": 3}}
        candle = self.normalizer.candle(event, None)
        assert candle.close == 1.5 and candle.volume == 10.0


class TestOrderBook:
    def test_snapshot_replaces_both_sides(self) -> None:
        book = OrderBook(5)
        snapshot = book.apply(
            DepthUpdate(
                ts=1.0,
                bids=(DepthLevel(99, 1),),
                asks=(DepthLevel(101, 2),),
                is_snapshot=True,
            )
        )
        assert snapshot.best_bid == 99.0 and snapshot.best_ask == 101.0

    def test_diff_only_touches_its_levels(self) -> None:
        book = OrderBook(5)
        book.apply(DepthUpdate(1.0, (DepthLevel(99, 1),), (DepthLevel(101, 2),), True))
        snapshot = book.apply(DepthUpdate(2.0, (DepthLevel(98, 5),), ()))
        assert snapshot.bid_depth(5) == 6.0
        assert snapshot.ask_depth(5) == 2.0

    def test_zero_size_removes_a_level(self) -> None:
        book = OrderBook(5)
        book.apply(DepthUpdate(1.0, (DepthLevel(99, 1),), (), True))
        snapshot = book.apply(DepthUpdate(2.0, (DepthLevel(99, 0),), ()))
        assert snapshot.bids == ()

    def test_trims_to_configured_depth(self) -> None:
        book = OrderBook(2)
        book.apply(
            DepthUpdate(
                1.0,
                tuple(DepthLevel(price, 1) for price in (99, 98, 97)),
                (),
                True,
            )
        )
        assert len(book.snapshot().bids) == 2

    def test_snapshot_sorts_sides(self) -> None:
        book = OrderBook(5)
        snapshot = book.apply(
            DepthUpdate(
                1.0,
                (DepthLevel(97, 1), DepthLevel(99, 1)),
                (DepthLevel(103, 1), DepthLevel(101, 1)),
                True,
            )
        )
        assert [level.price for level in snapshot.bids] == [99.0, 97.0]
        assert [level.price for level in snapshot.asks] == [101.0, 103.0]


class TestCandleBuilder:
    def test_closes_a_bar_when_the_timeframe_rolls_over(self) -> None:
        builder = CandleBuilder()
        first = make_trade(100.0, ts=DEFAULT_START)
        later = make_trade(101.0, ts=DEFAULT_START + 61)

        assert builder.push(first) is None
        closed = builder.push(later)
        assert closed is not None
        assert closed.close == pytest.approx(100.0)
        assert builder.current is not None
        assert builder.current.close == pytest.approx(101.0)

    def test_splits_volume_by_aggressor_side(self) -> None:
        builder = CandleBuilder()
        for trade in make_trades([(100.0, 2.0, Side.BUY), (100.0, 1.0, Side.SELL)]):
            builder.push(trade)
        bar = builder.flush()
        assert bar is not None
        assert bar.buy_volume == pytest.approx(2.0)
        assert bar.sell_volume == pytest.approx(1.0)
        assert bar.delta == pytest.approx(1.0)
        assert bar.levels[100.0].bid_trades == 1

    def test_drops_empty_bars_on_flush(self) -> None:
        assert CandleBuilder().flush() is None

    def test_records_symbol_and_timeframe(self) -> None:
        builder = CandleBuilder()
        builder.push(make_trade(100.0, symbol="ETHUSDT"))
        bar = builder.flush()
        assert bar is not None
        assert bar.symbol == "ETHUSDT"
        assert str(bar.timeframe) == "1m"


class TestFeeds:
    async def test_synthetic_feed_is_deterministic(self) -> None:
        first = [trade async for trade in SyntheticFeed(bars=3, seed=42).stream()]
        second = [trade async for trade in SyntheticFeed(bars=3, seed=42).stream()]
        assert [(t.ts, t.price, t.side) for t in first] == [(t.ts, t.price, t.side) for t in second]

    async def test_synthetic_feed_covers_the_requested_bars(self) -> None:
        trades = [trade async for trade in SyntheticFeed(bars=4, trades_per_bar=5).stream()]
        assert len(trades) == 20

    async def test_synthetic_feed_publishes_a_book(self) -> None:
        feed = SyntheticFeed(bars=1, trades_per_bar=2)
        async for _ in feed.stream():
            pass
        assert feed.snapshot().best_bid is not None

    async def test_replay_feed_round_trips_recorded_ticks(self, recorded_file) -> None:
        feed = ReplayFeed(recorded_file, symbol="BTCUSDT")
        trades = [trade async for trade in feed.stream()]
        assert trades
        assert all(isinstance(trade, Trade) for trade in trades)

    async def test_replay_feed_honours_the_time_window(self, recorded_file) -> None:
        full = [t.ts async for t in ReplayFeed(recorded_file).stream()]
        start = full[len(full) // 2]
        windowed = [trade async for trade in ReplayFeed(recorded_file, start=start).stream()]
        assert windowed
        assert len(windowed) < len(full)
        assert min(trade.ts for trade in windowed) >= start

    async def test_replay_feed_reports_missing_files(self) -> None:
        with pytest.raises(FileNotFoundError):
            [trade async for trade in ReplayFeed("missing.jsonl").stream()]

    def test_unknown_mode_is_rejected(self) -> None:
        config = make_config(mode="carrier-pigeon")
        with pytest.raises(ValueError, match="Unknown pipeline mode"):
            build_trade_feed(config)

    def test_replay_mode_requires_a_path(self) -> None:
        config = make_config(mode="replay")
        with pytest.raises(ValueError, match="replay_path"):
            build_trade_feed(config)


class TestIngestionHub:
    async def test_produces_bars_and_a_book(self, config: OrderFlowConfig) -> None:
        hub = IngestionHub(config)
        await hub.start()
        builder = CandleBuilder(config.timeframe, config.exchange.price_precision)
        bars = []
        async for trade in hub.trades():
            closed = builder.push(trade)
            if closed is not None:
                bars.append(closed)
        if (final := builder.flush()) is not None:
            bars.append(final)
        await hub.aclose()

        assert bars
        assert hub.snapshot().best_bid is not None
        assert hub.diagnostics()["trade_feed"] == "synthetic"

    async def test_records_ticks_when_configured(self, tmp_path) -> None:
        config = make_config()
        record_path = tmp_path / "recorded.jsonl"
        object.__setattr__(config.pipeline, "record_path", str(record_path))

        hub = IngestionHub(config)
        await hub.start()
        async for _ in hub.trades():
            pass
        await hub.aclose()

        assert record_path.exists()
        assert record_path.read_text(encoding="utf-8").count("\n") > 0


class TestResampler:
    # A five minute aligned start, so ten one minute bars form two clean groups.
    aligned_start = DEFAULT_START - (DEFAULT_START % 300)

    def test_aggregates_base_bars_into_longer_timeframes(self) -> None:
        bars = [
            make_bar(open_ts=self.aligned_start + index * 60, levels={100.0 + index: (1.0, 2.0)})
            for index in range(10)
        ]
        candles = resample_bars(bars, Timeframe.M5)

        assert len(candles) == 2
        first = candles[0]
        assert first.open == pytest.approx(100.0)
        assert first.close == pytest.approx(104.0)
        assert first.high == pytest.approx(104.0)
        assert first.low == pytest.approx(100.0)
        assert first.volume == pytest.approx(15.0)

    def test_returns_nothing_for_an_empty_series(self) -> None:
        assert resample_bars([], Timeframe.M5) == []

    def test_ignores_a_single_incomplete_group(self) -> None:
        bars = [make_bar(open_ts=self.aligned_start)]
        assert resample_bars(bars, Timeframe.M5) == []