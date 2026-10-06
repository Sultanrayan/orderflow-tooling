"""Command line entry points behind the documented scripts.

The suite never touches the network: every command runs against the shipped
synthetic configuration or a recorded JSON lines file.
"""

from __future__ import annotations

import pytest

from orderflow_ai.cli import main


class TestParser:
    def test_help_lists_the_documented_commands(self, capsys) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        for command in ("live", "replay", "collect", "backtest"):
            assert command in out

    def test_requires_a_sub_command(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code == 2


class TestBacktest:
    def test_runs_over_the_default_synthetic_config(self, capsys) -> None:
        code = main(["backtest", "--config", "configs/default.yaml", "--bars", "5"])

        assert code == 0
        assert "bars=5" in capsys.readouterr().out

    def test_accepts_the_documented_start_and_end_window(self, capsys) -> None:
        code = main(
            [
                "backtest",
                "--config",
                "configs/default.yaml",
                "--start",
                "2024-01-01",
                "--end",
                "2024-10-01",
                "--bars",
                "3",
            ]
        )

        assert code == 0
        assert "bars=3" in capsys.readouterr().out

    def test_replays_a_recorded_file(self, recorded_file, capsys) -> None:
        code = main(["backtest", "--file", str(recorded_file), "--bars", "3"])

        assert code == 0
        assert "bars=3" in capsys.readouterr().out

    def test_reports_a_missing_replay_file(self, capsys) -> None:
        code = main(["backtest", "--file", "does-not-exist.jsonl"])

        assert code == 1
        assert "Replay file not found" in capsys.readouterr().err

    def test_replay_mode_without_a_source_is_rejected(self, tmp_path, capsys) -> None:
        config = tmp_path / "replay.yaml"
        config.write_text("pipeline:\n  mode: replay\n", encoding="utf-8")

        code = main(["backtest", "--config", str(config)])

        assert code == 1
        assert "no replay file configured" in capsys.readouterr().err


class TestReplay:
    def test_analyses_a_recorded_file(self, recorded_file, capsys) -> None:
        code = main(["replay", "--file", str(recorded_file), "--bars", "3"])

        assert code == 0
        assert "BTCUSDT" in capsys.readouterr().out

    def test_reports_an_empty_recording(self, tmp_path, capsys) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")

        code = main(["replay", "--file", str(empty)])

        assert code == 1
        assert "no bars closed" in capsys.readouterr().err
