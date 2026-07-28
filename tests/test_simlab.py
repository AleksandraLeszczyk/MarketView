"""Tests for the SimLab simulation suite (clock, store, market, engine, scores)."""
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent_stonks import clock
from agent_stonks.agent import MOMENTUM_SYSTEM_PROMPT
from simlab import data as sim_data
from simlab import prompts as sim_prompts
from simlab import results as sim_results
from simlab.engine import SimulationConfig, SimulationEngine
from simlab.judge import _entry_context, _first_exit_after
from simlab.market import SimMarket
from simlab.patches import simulation_context

DAY = date(2026, 6, 15)  # a Monday
OPEN_UTC = datetime(2026, 6, 15, 13, 30, tzinfo=timezone.utc)  # 09:30 EDT


def _bar(ts: datetime, close: float, volume: float = 1000.0) -> dict:
    return {
        "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "o": close - 0.05,
        "h": close + 0.1,
        "l": close - 0.1,
        "c": close,
        "v": volume,
    }


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the SimLab store at a temp dir and populate one synthetic day:
    60 minute bars ramping 100.0 -> 105.9, plus 30 prior daily bars."""
    monkeypatch.setattr(sim_data, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(sim_data, "MANIFEST_PATH", tmp_path / "datasets.json")
    minute_bars = [
        _bar(OPEN_UTC + timedelta(minutes=i), 100.0 + 0.1 * i) for i in range(60)
    ]
    sim_data._write_gz(sim_data.bars_path("TEST", DAY), minute_bars)
    daily = [
        _bar(datetime(2026, 6, 15, tzinfo=timezone.utc) - timedelta(days=i), 99.0)
        for i in range(30, 0, -1)
    ]
    sim_data._write_gz(sim_data.daily_path("TEST"), {
        "symbol": "TEST", "start": "2026-05-16", "end": "2026-06-15", "bars": daily,
    })
    return tmp_path


class TestClock:
    def test_pin_and_clear(self):
        pinned = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)
        try:
            clock.set_simulated(pinned)
            assert clock.now() == pinned
            assert clock.monotonic() == pinned.timestamp()
            assert clock.is_simulated()
        finally:
            clock.clear()
        assert not clock.is_simulated()
        assert abs((clock.now() - datetime.now(timezone.utc)).total_seconds()) < 5

    def test_rejects_naive_datetime(self):
        with pytest.raises(ValueError):
            clock.set_simulated(datetime(2026, 6, 15, 14, 0))


class TestMarket:
    def test_completed_bars_respect_bar_completion(self, store):
        market = SimMarket(["TEST"], [DAY])
        # At 13:31:00 exactly bar 0 (13:30) has completed; bar 1 has not.
        t = OPEN_UTC + timedelta(minutes=1)
        bars = market.completed_bars("TEST", t)
        assert len(bars) == 1
        assert market.price_at("TEST", t) == 100.0

    def test_daily_bars_include_partial_today(self, store):
        market = SimMarket(["TEST"], [DAY])
        t = OPEN_UTC + timedelta(minutes=10)
        daily = market.daily_bars_at("TEST", t)
        assert daily[-1]["t"].startswith("2026-06-15")
        assert daily[-1]["c"] == pytest.approx(100.9)  # close of bar 9
        assert market.prev_close("TEST", t) == 99.0

    def test_step_times_cover_every_bar(self, store):
        market = SimMarket(["TEST"], [DAY])
        steps = market.step_times(DAY)
        assert len(steps) == 60
        assert steps[0] == OPEN_UTC + timedelta(minutes=1)


class TestDatasetStore:
    def test_create_dataset_downloads_only_missing_days(self, store, monkeypatch):
        calls = []

        def fake_minute(symbol, day, key, secret, feed="iex"):
            calls.append(day)
            return [_bar(OPEN_UTC, 100.0)] if day.weekday() < 5 else []

        monkeypatch.setattr(sim_data, "fetch_minute_bars_day", fake_minute)
        monkeypatch.setattr(sim_data, "fetch_news_day", lambda *a, **k: [])
        monkeypatch.setattr(sim_data, "fetch_daily_bars_range", lambda *a, **k: [_bar(OPEN_UTC, 99.0)])
        monkeypatch.setattr(sim_data, "fetch_market_indicator_closes", lambda *a, **k: {"spy": [], "vix": [], "vix3m": []})

        ds = sim_data.create_dataset("wk", ["TEST"], date(2026, 6, 15), date(2026, 6, 17), "k", "s")
        # 2026-06-15 already in the store (fixture) -- only 16th and 17th fetched.
        assert calls == [date(2026, 6, 16), date(2026, 6, 17)]
        assert ds.days == ["2026-06-15", "2026-06-16", "2026-06-17"]
        assert sim_data.get_dataset("wk").symbols == ["TEST"]
        sim_data.delete_dataset("wk")
        assert sim_data.get_dataset("wk") is None


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def _response(tool_calls, content=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, model, messages, tools, tool_choice):
                outer.calls.append(messages)
                return outer._responses.pop(0)

        self.chat = SimpleNamespace(completions=_Completions())


def _scripted_client():
    """Cycle 1: arm a buy-the-breakout tactic at 103 and finalize with alert.
    Cycle 2 (woken by the fill): sell everything. Cycle 3 (cycle timer): stand
    aside on a far-away alert for the rest of the day."""
    cycle1 = _response([
        _tool_call("c1a", "set_tactics", {
            "symbol": "TEST",
            "actions": [{
                "action": "buy", "quantity": 10,
                "conditions": [{"field": "last_price", "condition": "above", "value": 103.0}],
                "note": "breakout entry",
            }],
            "reasoning": "buy the break of 103",
        }),
        _tool_call("c1b", "submit_decision", {
            "action": "alert", "regime": "bullish", "reasoning": "waiting for the break", "alerts": [],
        }),
    ])
    cycle2 = _response([
        _tool_call("c2a", "submit_decision", {
            "action": "sell", "symbol": "TEST", "quantity": 10,
            "regime": "bullish", "reasoning": "taking the breakout profit",
        }),
    ])
    cycle3 = _response([
        _tool_call("c3a", "submit_decision", {
            "action": "alert", "regime": "neutral", "reasoning": "nothing to do",
            "alerts": [{"symbol": "TEST", "field": "last_price", "condition": "above", "value": 99999}],
        }),
    ])
    return FakeClient([cycle1, cycle2, cycle3])


def _run_sim(store, cycle_minutes=5):
    market = SimMarket(["TEST"], [DAY])
    config = SimulationConfig(
        personality="momentum", provider="openai", model="fake", api_key="",
        symbols=["TEST"], days=[DAY], starting_cash=10_000.0, cycle_minutes=cycle_minutes,
    )
    engine = SimulationEngine(market, config)
    result = engine.run(client=_scripted_client())
    return market, result


class TestEngine:
    def test_full_session_replay(self, store):
        market, result = _run_sim(store)
        assert result.error is None
        assert result.cycles_run == 3

        actions = [(d["action"], d["status"]) for d in result.decisions]
        assert actions == [
            ("tactics", "armed"),  # cycle 1: arm the breakout entry
            ("alert", "noop"),  # cycle 1: finalize (empty alerts, tactics armed)
            ("buy", "filled"),  # tactic fires mid-fast-forward
            ("sell", "filled"),  # cycle 2, woken by the fill
            ("alert", "noop"),  # cycle 3, cycle timer
        ]
        buy, sell = result.decisions[2], result.decisions[3]
        # The tactic fired on the first bar closing >= 103 (bar 30, completes
        # 14:01Z) and filled at that bar's close -- simulated time throughout.
        assert buy["price"] == pytest.approx(103.0)
        assert buy["ts"].startswith("2026-06-15T14:01")
        assert "Tactics triggered" in buy["reasoning"]
        assert sell["price"] >= buy["price"]  # ramping tape: sold at/above the entry

        # Equity: one point per completed bar, valued in simulated time.
        assert len(result.equity) == 60
        assert result.equity[0]["ts"].startswith("2026-06-15")
        fees = 2 * 1.15
        expected_final = 10_000.0 + 10 * (sell["price"] - buy["price"]) - fees
        assert result.final_value == pytest.approx(expected_final)

    def test_clock_restored_after_run(self, store):
        _run_sim(store)
        assert not clock.is_simulated()

    def test_result_records_the_prompt_and_tools_that_produced_it(self, store):
        # `prompt_overridden` alone can't distinguish two revisions of the
        # built-in prompt, so the run has to carry the resolved text itself.
        _, result = _run_sim(store)
        assert result.prompt_used == MOMENTUM_SYSTEM_PROMPT
        assert "get_quote" in result.tool_names
        assert "analyze_swing_levels" in result.tool_names

    def test_result_records_an_overriding_prompt(self, store):
        market = SimMarket(["TEST"], [DAY])
        config = SimulationConfig(
            personality="momentum", provider="openai", model="fake", api_key="",
            symbols=["TEST"], days=[DAY], starting_cash=10_000.0, cycle_minutes=5,
            system_prompt_override="custom plan",
        )
        result = SimulationEngine(market, config).run(client=_scripted_client())
        assert result.prompt_used == "custom plan"

    def test_summary_and_oracle(self, store):
        market, result = _run_sim(store)
        summary = sim_results.summarize_run(result, market)
        assert summary["trades_filled"] == 2
        # Oracle: buy 100.0, sell 105.9 -> 5.9%.
        assert summary["oracle_ceiling_pct"] == pytest.approx(5.9, abs=0.01)
        assert summary["profit_efficiency"] is not None
        assert summary["return_pct"] == pytest.approx(
            (result.final_value / 10_000.0 - 1.0) * 100.0
        )


class TestOracle:
    def test_best_round_trip_orders_matter(self):
        assert sim_results.oracle_best_round_trip([105, 100, 104]) == pytest.approx(4.0)
        assert sim_results.oracle_best_round_trip([105, 104, 103]) == 0.0
        assert sim_results.oracle_best_round_trip([]) == 0.0


class TestJudgeContext:
    def test_entry_context_includes_tape_and_outcome(self, store):
        market, result = _run_sim(store)
        buy = result.decisions[2]
        exit_decision = _first_exit_after(buy, result.decisions)
        assert exit_decision is not None and exit_decision["action"] == "sell"
        context = _entry_context(buy, exit_decision, market)
        assert "ENTRY: buy" in context
        assert "TAPE BEFORE ENTRY" in context
        assert "max favorable excursion" in context
        assert "EXIT: sold" in context


class TestPatches:
    def test_market_indicators_clip_to_sim_time(self, store):
        sim_data._write_gz(sim_data.market_path(), {
            "spy": [
                {"date": "2026-06-12", "close": 500.0},
                {"date": "2026-06-15", "close": 501.0},
                {"date": "2026-06-16", "close": 502.0},
            ],
            "vix": [{"date": "2026-06-15", "close": 15.0}],
            "vix3m": [],
        })
        market = SimMarket(["TEST"], [DAY])
        from agent_stonks import historical

        with simulation_context(market):
            clock.set_simulated(OPEN_UTC)
            series = historical.fetch_market_indicators()
            # 2026-06-16 is the future from the pinned clock -- must be clipped.
            assert list(series["spy"].values) == [500.0, 501.0]
            assert float(series["vix"].iloc[-1]) == 15.0
        assert not clock.is_simulated()

    def test_patches_are_restored(self, store):
        from agent_stonks import historical

        original = historical.fetch_market_indicators
        with simulation_context(SimMarket(["TEST"], [DAY])):
            assert historical.fetch_market_indicators is not original
        assert historical.fetch_market_indicators is original


class TestPrompts:
    def test_override_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_prompts, "PROMPTS_DIR", tmp_path / "prompts")
        assert sim_prompts.get_prompt("momentum") == sim_prompts.default_prompt("momentum")
        assert not sim_prompts.has_override("momentum")
        sim_prompts.save_override("momentum", "You are a test agent.")
        assert sim_prompts.get_prompt("momentum") == "You are a test agent."
        sim_prompts.reset_override("momentum")
        assert sim_prompts.get_prompt("momentum") == sim_prompts.default_prompt("momentum")

    def test_unknown_personality_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_prompts, "PROMPTS_DIR", tmp_path / "prompts")
        with pytest.raises(KeyError):
            sim_prompts.save_override("nope", "x")


class TestExperiments:
    @pytest.fixture(autouse=True)
    def exp_store(self, tmp_path, monkeypatch):
        from simlab import experiments as sim_experiments

        monkeypatch.setattr(sim_experiments, "EXPERIMENTS_DIR", tmp_path / "experiments")
        self.experiments = sim_experiments

    def _submit(self, **overrides):
        config = {
            "personality": "momentum", "provider": "openai", "model": "gpt-test",
            "api_key": "sk-secret", "symbols": ["TEST"], "days": [DAY.isoformat()],
            "starting_cash": 100_000.0, "cycle_minutes": 5,
            "max_cycles_per_day": 40, "system_prompt_override": None,
            "run_judge": False,
        }
        config.update(overrides)
        return self.experiments.submit("ds1", config)

    def test_submit_and_finalize_scrubs_api_key(self):
        exp = self._submit(judge_api_key="sk-judge")
        assert exp["status"] == self.experiments.WAITING
        stored = self.experiments.get_experiment(exp["experiment_id"])
        assert stored["config"]["api_key"] == "sk-secret"

        done = self.experiments.finalize(
            exp["experiment_id"], self.experiments.FINISHED, run_id="run-1",
            result_summary={"return_pct": 1.5},
        )
        assert done["status"] == self.experiments.FINISHED
        assert done["run_id"] == "run-1"
        assert done["config"]["api_key"] == ""
        assert done["config"]["judge_api_key"] == ""
        assert done["finished_at"] is not None

    def test_tick_spawns_up_to_limit_oldest_first(self, monkeypatch):
        first = self._submit()
        second = self._submit()
        third = self._submit()
        spawned = []
        monkeypatch.setattr(self.experiments, "spawn", spawned.append)
        self.experiments.tick(max_parallel=2)
        assert spawned == [first["experiment_id"], second["experiment_id"]]
        assert third["experiment_id"] not in spawned

    def test_tick_counts_running_against_limit(self, monkeypatch):
        running = self._submit()
        self.experiments.update(
            running["experiment_id"], status=self.experiments.RUNNING, pid=99999999
        )
        waiting = self._submit()
        spawned = []
        monkeypatch.setattr(self.experiments, "spawn", spawned.append)
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: True)
        self.experiments.tick(max_parallel=1)
        assert spawned == []
        self.experiments.tick(max_parallel=2)
        assert spawned == [waiting["experiment_id"]]

    def test_stop_kills_the_worker_and_finalizes(self, monkeypatch):
        exp = self._submit()
        self.experiments.update(
            exp["experiment_id"], status=self.experiments.RUNNING, pid=4242
        )
        killed = []
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: True)
        monkeypatch.setattr(self.experiments.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(
            self.experiments.os, "killpg", lambda pgid, sig: killed.append((pgid, sig))
        )
        stopped = self.experiments.stop(exp["experiment_id"])
        assert killed == [(4242, self.experiments.signal.SIGTERM)]
        assert stopped["status"] == self.experiments.FAILED
        assert stopped["error"] == self.experiments.STOPPED_ERROR
        assert stopped["config"]["api_key"] == ""

    def test_stop_leaves_a_finished_experiment_alone(self, monkeypatch):
        exp = self._submit()
        self.experiments.finalize(
            exp["experiment_id"], self.experiments.FINISHED, run_id="run-1"
        )
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: True)
        untouched = self.experiments.stop(exp["experiment_id"])
        assert untouched["status"] == self.experiments.FINISHED
        assert untouched["run_id"] == "run-1"
        assert untouched["error"] is None

    def test_stop_finalizes_a_waiting_experiment_without_a_worker(self):
        exp = self._submit()
        stopped = self.experiments.stop(exp["experiment_id"])
        assert stopped["status"] == self.experiments.FAILED
        assert stopped["error"] == self.experiments.STOPPED_ERROR

    def test_stopped_experiment_is_not_reaped_again(self, monkeypatch):
        exp = self._submit()
        self.experiments.update(
            exp["experiment_id"], status=self.experiments.RUNNING, pid=4242
        )
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(self.experiments, "spawn", lambda _id: None)
        self.experiments.stop(exp["experiment_id"])
        self.experiments.tick(max_parallel=1)
        after = self.experiments.get_experiment(exp["experiment_id"])
        assert after["error"] == self.experiments.STOPPED_ERROR

    def test_tick_reaps_dead_workers(self, monkeypatch):
        exp = self._submit()
        self.experiments.update(
            exp["experiment_id"], status=self.experiments.RUNNING, pid=99999999
        )
        monkeypatch.setattr(self.experiments, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(self.experiments, "spawn", lambda _id: None)
        self.experiments.tick(max_parallel=1)
        reaped = self.experiments.get_experiment(exp["experiment_id"])
        assert reaped["status"] == self.experiments.FAILED
        assert "died" in reaped["error"]

    def test_clear_finished_keeps_active(self):
        active = self._submit()
        done = self._submit()
        self.experiments.finalize(done["experiment_id"], self.experiments.FINISHED)
        assert self.experiments.clear_finished() == 1
        ids = [e["experiment_id"] for e in self.experiments.list_experiments()]
        assert ids == [active["experiment_id"]]


class TestBreakdown:
    def _run(self, model="gpt-a", dataset="ds1", personality="momentum",
             return_pct=1.0, efficiency=0.5, judge_score=7.0):
        record = {
            "dataset": dataset,
            "config_summary": {"provider": "openai", "model": model, "personality": personality},
            "summary": {"return_pct": return_pct, "profit_efficiency": efficiency},
        }
        if judge_score is not None:
            record["judge"] = {"overall_score": judge_score}
        return record

    def test_groups_by_model(self):
        rows = sim_results.breakdown(
            [
                self._run(model="gpt-a", return_pct=2.0, efficiency=0.8, judge_score=8.0),
                self._run(model="gpt-a", return_pct=0.0, efficiency=0.2, judge_score=None),
                self._run(model="gpt-b", return_pct=-1.0, efficiency=0.1),
            ],
            by="model",
        )
        assert [r["group"] for r in rows] == ["openai/gpt-a", "openai/gpt-b"]
        top = rows[0]
        assert top["runs"] == 2
        assert top["avg_return_pct"] == 1.0
        assert top["best_return_pct"] == 2.0
        assert top["avg_profit_efficiency"] == 0.5
        assert top["avg_judge_score"] == 8.0  # unjudged run skipped

    def test_groups_by_dataset_and_agent(self):
        runs = [
            self._run(dataset="ds1", personality="momentum"),
            self._run(dataset="ds2", personality="contrarian", efficiency=None),
        ]
        by_dataset = sim_results.breakdown(runs, by="dataset")
        assert {r["group"] for r in by_dataset} == {"ds1", "ds2"}
        by_agent = sim_results.breakdown(runs, by="agent")
        assert {r["group"] for r in by_agent} == {"momentum", "contrarian"}
        no_eff = next(r for r in by_agent if r["group"] == "contrarian")
        assert no_eff["avg_profit_efficiency"] is None
        # None-efficiency groups sort after scored ones.
        assert by_agent[-1]["group"] == "contrarian"

    def test_best_run_names_the_winning_run(self):
        rows = sim_results.breakdown(
            [
                self._run(model="gpt-a", dataset="ds1", personality="momentum", return_pct=2.0),
                self._run(model="gpt-a", dataset="ds2", personality="breakout", return_pct=5.0),
            ],
            by="model",
        )
        best = rows[0]["best_run"]
        assert rows[0]["best_return_pct"] == 5.0
        assert best["return_pct"] == 5.0
        assert (best["provider"], best["model"]) == ("openai", "gpt-a")
        assert best["personality"] == "breakout"
        assert best["dataset"] == "ds2"

    def test_best_run_absent_without_returns(self):
        rows = sim_results.breakdown([self._run(return_pct=None)], by="model")
        assert rows[0]["best_return_pct"] is None
        assert rows[0]["best_run"] is None

    def test_unknown_dimension_rejected(self):
        with pytest.raises(ValueError):
            sim_results.breakdown([], by="provider-only")


class TestPriorRuns:
    def _run(self, run_id="r1", personality="momentum", provider="openai",
             model="gpt-a", dataset="ds1", agent_log=None, **fields):
        record = {
            "run_id": run_id,
            "dataset": dataset,
            "config_summary": {
                "personality": personality, "provider": provider, "model": model,
                "days": [DAY.isoformat()],
            },
            "summary": {"return_pct": 1.0, "profit_efficiency": 0.5},
            "cycles_run": 10,
            "agent_log": agent_log if agent_log is not None else [{"type": "cycle"}],
        }
        record.update(fields)
        return record

    def test_counts_only_error_log_entries(self):
        record = self._run(agent_log=[
            {"type": "cycle"},
            {"type": "error", "text": "LLM call failed: Error code: 429"},
            {"type": "error", "text": "Agent cycle failed: boom"},
        ])
        assert sim_results.cycle_error_count(record) == 2
        assert sim_results.cycle_error_count(self._run()) == 0
        assert sim_results.cycle_error_count({}) == 0

    def test_matches_only_the_full_combination(self):
        runs = [
            self._run(run_id="match"),
            self._run(run_id="other-agent", personality="breakout"),
            self._run(run_id="other-model", model="gpt-b"),
            self._run(run_id="other-provider", provider="anthropic"),
            self._run(run_id="other-dataset", dataset="ds2"),
        ]
        prior = sim_results.find_prior_runs(runs, "momentum", "openai", "gpt-a", "ds1")
        assert [r["run_id"] for r in prior["clean"]] == ["match"]
        assert prior["degraded"] == []

    def test_llm_errors_make_a_run_degraded_not_clean(self):
        runs = [
            self._run(run_id="clean"),
            self._run(run_id="llm-error", agent_log=[
                {"type": "error", "text": "LLM call failed: Error code: 429"}]),
            self._run(run_id="engine-error", error="no simulated tape price"),
            self._run(run_id="interrupted", interrupted=True),
            self._run(run_id="no-cycles", cycles_run=0),
        ]
        prior = sim_results.find_prior_runs(runs, "momentum", "openai", "gpt-a", "ds1")
        assert [r["run_id"] for r in prior["clean"]] == ["clean"]
        assert [r["run_id"] for r in prior["degraded"]] == [
            "llm-error", "engine-error", "interrupted", "no-cycles"
        ]

    def test_no_prior_runs(self):
        prior = sim_results.find_prior_runs([], "momentum", "openai", "gpt-a", "ds1")
        assert prior == {"clean": [], "degraded": []}

    def test_store_signature_tracks_changes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sim_results, "RUNS_DIR", tmp_path / "runs")
        assert sim_results.store_signature() == ()
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "a.json").write_text("{}")
        first = sim_results.store_signature()
        assert len(first) == 1
        (tmp_path / "runs" / "b.json").write_text("{}")
        assert sim_results.store_signature() != first


class TestRunnerJudgeSelection:
    """The judge LLM is picked independently of the agent's, falling back to it."""

    def _run_with(self, monkeypatch, config: dict) -> dict:
        from simlab import runner

        captured: dict = {}

        def fake_judge_run(decisions, summary, prompt, market, provider, api_key,
                           model, progress=lambda msg: None):
            captured.update(provider=provider, api_key=api_key, model=model)
            return {}

        class FakeEngine:
            def __init__(self, *a, **kw):
                pass

            def run(self):
                return SimpleNamespace(cycles_run=1, decisions=[])

        monkeypatch.setattr(
            runner.experiments, "get_experiment",
            lambda _id: {"config": config, "dataset": "ds1"},
        )
        monkeypatch.setattr(runner, "SimMarket", lambda *a, **kw: object())
        monkeypatch.setattr(runner, "SimulationEngine", FakeEngine)
        monkeypatch.setattr(runner.sim_results, "summarize_run", lambda *a, **kw: {})
        monkeypatch.setattr(runner.sim_results, "save_run", lambda *a, **kw: {"run_id": "r1"})
        monkeypatch.setattr(runner.sim_prompts, "get_prompt", lambda _p: "brief")
        monkeypatch.setattr(runner.sim_judge, "judge_run", fake_judge_run)
        runner.run_experiment("exp-1")
        return captured

    @staticmethod
    def _config(**overrides) -> dict:
        config = {
            "personality": "momentum", "provider": "openai", "model": "gpt-test",
            "api_key": "sk-agent", "symbols": ["TEST"], "days": [DAY.isoformat()],
            "starting_cash": 100_000.0, "cycle_minutes": 5,
            "max_cycles_per_day": 40, "system_prompt_override": None,
            "run_judge": True,
        }
        config.update(overrides)
        return config

    def test_separate_judge_llm_is_used(self, monkeypatch):
        captured = self._run_with(monkeypatch, self._config(
            judge_provider="anthropic", judge_model="claude-sonnet-5",
            judge_api_key="sk-judge",
        ))
        assert captured == {
            "provider": "anthropic", "model": "claude-sonnet-5", "api_key": "sk-judge",
        }

    def test_falls_back_to_the_agent_llm(self, monkeypatch):
        """Experiments queued before the judge picker existed still judge."""
        captured = self._run_with(monkeypatch, self._config())
        assert captured == {
            "provider": "openai", "model": "gpt-test", "api_key": "sk-agent",
        }
