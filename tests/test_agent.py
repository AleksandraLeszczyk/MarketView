import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent_stonks import historical as agent_historical
from agent_stonks import market_hours
from agent_stonks.agent import (
    BREAKOUT_TOOLS,
    MOMENTUM_SYSTEM_PROMPT,
    MOMENTUM_TOOLS,
    PERSONALITY_TOOLS,
    REVERSAL_TOOLS,
    SMART_MONEY_TOOLS,
    VOLUME_DETECTIVE_SYSTEM_PROMPT,
    VOLUME_DETECTIVE_TOOLS,
    _dispatch_tool,
    _session_closed_addendum,
    _tool_analyze_consolidation,
    _tool_analyze_volume,
    _tool_analyze_volume_profile_2,
    _tool_breakout_trade_geometry,
    _tool_get_corporate_actions,
    _tool_get_news,
    _tool_get_quote,
    _tool_smart_money_trade_geometry,
    _tool_vwap_reversion_geometry,
    _wait_for_next_cycle,
    run_agent_cycle,
)
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState, alert_triggered


def _app(*symbols: str):
    """AppState streaming `symbols` (default AAPL) plus its first SymbolState."""
    app = AppState()
    app.set_symbols(list(symbols) or ["AAPL"])
    return app, app.sym(app.symbols[0])


class FakeBroker(Broker):
    def __init__(self, price: float = 100.0):
        self.price = price

    def get_current_price(self, symbol, key, secret, feed="iex") -> float:
        return self.price

    def submit_order(self, symbol, side, quantity, price) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Returns canned chat-completion responses in sequence."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list = []
        self.tools_seen: list = []
        outer = self

        class _Completions:
            def create(self, model, messages, tools, tool_choice):
                outer.calls.append(messages)
                outer.tools_seen.append(tools)
                return outer._responses.pop(0)

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        self.chat = _Chat()


class TestToolHandlers:
    def test_analyze_volume_with_no_bars_returns_note(self, monkeypatch):
        # With no Alpaca bars and yfinance volume unavailable, the tool reports
        # the honest "no bars" note rather than fabricating a read.
        monkeypatch.setattr(agent_historical, "fetch_intraday_volume_bars", lambda symbol: [])
        _, state = _app()
        assert "note" in _tool_analyze_volume(state)

    def test_analyze_volume_uses_yfinance_consolidated_volume(self, monkeypatch):
        # Volume is sourced from yfinance (consolidated tape) and reported as a
        # full-tape feed, not Alpaca's single-venue IEX partial feed.
        _, state = _app()
        bars = [
            {"t": f"2026-07-21T14:{m:02d}:00Z", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 10_000 + m}
            for m in range(30)
        ]
        monkeypatch.setattr(agent_historical, "fetch_intraday_volume_bars", lambda symbol: bars)
        monkeypatch.setattr(agent_historical, "fetch_daily_volume_bars", lambda symbol: [])
        result = _tool_analyze_volume(state)
        assert result.get("partial_volume_feed") is None

    def test_analyze_volume_profile_2_no_bars_returns_note(self, monkeypatch):
        # No yfinance intraday and no live buffer -> honest note, no fabrication.
        monkeypatch.setattr(agent_historical, "fetch_intraday_volume_bars", lambda symbol: [])
        _, state = _app()
        assert "note" in _tool_analyze_volume_profile_2(state)

    def test_analyze_volume_profile_2_reads_yfinance_and_classifies(self, monkeypatch):
        # Today's live path sources 1-min bars from yfinance and returns the
        # peaks list with the documented shape.
        from datetime import timedelta

        base = datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc)  # 09:30 ET
        closes = [100.0] * 15 + [100.0 + 0.1 * i for i in range(1, 16)] + [
            103.0 - 0.1 * i for i in range(1, 16)
        ]
        vols = [5000] * 15 + [1000] * 30
        vols[30] = 6000
        bars = [
            {
                "t": (base + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "o": c, "h": c + 0.05, "l": c - 0.05, "c": c, "v": float(v),
            }
            for i, (c, v) in enumerate(zip(closes, vols))
        ]
        monkeypatch.setattr(agent_historical, "fetch_intraday_volume_bars", lambda symbol: bars)
        _, state = _app()
        state.last_price = 101.5
        result = _tool_analyze_volume_profile_2(state)
        assert result["peaks"]
        assert all(p["type"] in {"supply", "demand", "unsure", "scattered"} for p in result["peaks"])

    def test_get_quote_reads_state(self):
        _, state = _app()
        state.last_price = 101.0
        state.bid_price = 100.5
        result = _tool_get_quote(state)
        assert result["last_price"] == 101.0
        assert result["bid_price"] == 100.5

    def test_get_news_maps_impact_labels(self):
        _, state = _app()
        state.news = [{"id": "1", "headline": "h", "summary": "s", "created_at": "t", "source": "src"}]
        state.news_impacts = {"1": "positive"}
        result = _tool_get_news(state)
        assert result["articles"][0]["impact"] == "positive"

    def test_get_corporate_actions_without_keys_returns_note(self):
        _, state = _app()
        result = _tool_get_corporate_actions(state)
        assert "note" in result

    def test_get_corporate_actions_fetches_and_clamps_window(self, requests_mock):
        app, state = _app()
        app.api_key = "k"
        app.api_secret = "s"
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/corporate-actions",
            json={
                "corporate_actions": {
                    "cash_dividends": [{"symbol": "AAPL", "rate": 0.25, "ex_date": "2026-07-20"}]
                }
            },
        )
        result = _dispatch_tool("get_corporate_actions", {"days_ahead": 500}, app, DecisionTracker())
        assert result["window_days"] == 90
        assert result["upcoming_corporate_actions"][0]["type"] == "cash_dividend"

    def test_get_corporate_actions_empty_returns_note(self, requests_mock):
        app, state = _app()
        app.api_key = "k"
        app.api_secret = "s"
        requests_mock.get(
            "https://data.alpaca.markets/v1beta1/corporate-actions",
            json={"corporate_actions": {}},
        )
        result = _dispatch_tool("get_corporate_actions", {}, app, DecisionTracker())
        assert "no corporate actions" in result["note"]

    def test_dispatch_unknown_tool_returns_error(self):
        state = AppState()
        tracker = DecisionTracker()
        result = _dispatch_tool("nonexistent", {}, state, tracker)
        assert "error" in result

    def test_breakout_trade_geometry_tool_computes_targets(self):
        result = _tool_breakout_trade_geometry(entry=100.0, stop=98.0, atr=4.0)
        assert result["meets_min_reward_risk"] is True

    def test_breakout_personality_uses_breakout_tools(self):
        assert PERSONALITY_TOOLS["breakout"] is BREAKOUT_TOOLS
        names = {t["function"]["name"] for t in BREAKOUT_TOOLS}
        # get_key_levels feeds breakout_trade_geometry's overhead_resistance;
        # get_session_clock gates the timing windows; analyze_market sets the
        # broad risk backdrop.
        assert {
            "analyze_opening_range",
            "analyze_volume",
            "breakout_trade_geometry",
            "get_key_levels",
            "get_session_clock",
            "analyze_market",
        } <= names
        assert "analyze_daily_trend" not in names
        assert "get_put_call_walls" not in names

    def test_momentum_personality_uses_momentum_tools(self):
        assert PERSONALITY_TOOLS["momentum"] is MOMENTUM_TOOLS
        names = {t["function"]["name"] for t in MOMENTUM_TOOLS}
        assert {
            "analyze_intraday_momentum",
            "analyze_volume",
            "analyze_consolidation",
            "get_key_levels",
            "breakout_trade_geometry",
            "get_news",
            "get_quote",
        } <= names
        assert "analyze_daily_trend" not in names
        assert "analyze_market" not in names
        assert "analyze_opening_range" not in names
        assert "get_put_call_walls" not in names

    def test_analyze_consolidation_with_no_bars_returns_note(self):
        _, state = _app()
        assert "note" in _tool_analyze_consolidation(state)

    def test_get_key_levels_reads_intraday_and_daily_bars(self):
        app, state = _app()
        state.daily_bars = [{"t": "2026-07-16T04:00:00Z", "o": 100.0, "h": 110.0, "l": 95.0, "c": 105.0, "v": 1e6}]
        state.bars.append({"t": "2026-07-17T13:31:00Z", "o": 102.0, "h": 104.0, "l": 101.0, "c": 103.0, "v": 1000})
        state.last_price = 103.0
        result = _dispatch_tool("get_key_levels", {"symbol": "AAPL"}, app, DecisionTracker())
        assert result["levels"]["prior_day_high"] == 110.0
        assert result["levels"]["session_high"] == 104.0
        assert result["nearest_resistance"]["level"] == 104.0

    def test_breakout_trade_geometry_tool_flags_close_overhead_resistance(self):
        result = _tool_breakout_trade_geometry(entry=100.0, stop=98.0, atr=3.0, overhead_resistance=101.0)
        assert result["room_to_run"] is False

    def test_advanced_level_tools_dispatch_and_are_exposed_to_momentum_only(self):
        # Steps 4-6 of the S/R plan. Momentum step 3 asks for the first level
        # leaving 1.5 ATR of room rather than the nearest tick of session
        # structure, which needs these level sources; the other personalities
        # keep the narrower toolset -- except analyze_volume_profile_2, which
        # the Volume Signal Detective is built around.
        app, state = _app()
        tracker = DecisionTracker()
        assert "note" in _dispatch_tool("analyze_swing_levels", {"symbol": "AAPL"}, app, tracker)
        assert "note" in _dispatch_tool("analyze_volume_profile", {"symbol": "AAPL"}, app, tracker)
        state.daily_bars = [{"t": "2026-07-16T04:00:00Z", "o": 102.0, "h": 110.0, "l": 100.0, "c": 105.0, "v": 1e6}]
        state.last_price = 106.0
        pivots = _dispatch_tool("get_floor_pivots", {"symbol": "AAPL"}, app, tracker)
        assert pivots["levels"]["pivot"] == 105.0
        assert pivots["nearest_resistance"]["name"] == "r1"
        advanced = {"analyze_swing_levels", "analyze_volume_profile", "get_floor_pivots"}
        for personality, tools in PERSONALITY_TOOLS.items():
            names = {t["function"]["name"] for t in tools}
            if personality == "momentum":
                assert advanced <= names
            else:
                assert not advanced & names
            if personality != "volume_detective":
                assert "analyze_volume_profile_2" not in names

    def test_momentum_prompt_carries_the_advanced_levels_guidance(self):
        # The tools and the guidance that explains them ship together --
        # exposing one without the other is the failure mode worth catching.
        assert "ADVANCED LEVELS (confluence)" in MOMENTUM_SYSTEM_PROMPT
        assert "analyze_swing_levels" in MOMENTUM_SYSTEM_PROMPT

    def test_volume_detective_personality_uses_volume_detective_tools(self):
        assert PERSONALITY_TOOLS["volume_detective"] is VOLUME_DETECTIVE_TOOLS
        names = {t["function"]["name"] for t in VOLUME_DETECTIVE_TOOLS}
        # analyze_volume_profile_2 is the primary read; the rest cross-examine
        # its levels (structure, participation, approach, timing, catalyst) and
        # gate the geometry before any size.
        assert {
            "analyze_volume_profile_2",
            "detect_regime_shift",
            "get_quote",
            "analyze_volume",
            "analyze_intraday_momentum",
            "get_key_levels",
            "get_session_clock",
            "breakout_trade_geometry",
            "get_news",
            "get_position",
            "set_tactics",
            "submit_decision",
        } == names
        assert "analyze_daily_trend" not in names
        assert "get_put_call_walls" not in names

    def test_volume_detective_prompt_gates_levels_on_the_trajectory_read(self):
        # The tool and the guidance that makes it binding ship together: a
        # level map alone keeps bidding demand into a tape that has turned.
        prompt = VOLUME_DETECTIVE_SYSTEM_PROMPT
        assert "detect_regime_shift" in prompt
        assert "THE TRAJECTORY OVERRIDES THE LEVEL" in prompt
        # The trajectory read runs before the level map, not after it.
        assert prompt.index("READ THE TRAJECTORY FIRST") < prompt.index("MAP THE LEVELS")
        # Reacting while asleep: resting bids carry a momentum guard, and a
        # held position carries a trajectory-based exit.
        assert "REGIME EXIT" in prompt and "REGIME WAKE" in prompt
        assert "momentum_pct" in prompt

    def test_detect_regime_shift_dispatches_and_reads_live_bars(self):
        app, state = _app()
        # Live streamed buffer (not the delayed consolidated feed): 30 minutes
        # up, then a slide -- the turn the detective has to catch.
        base = datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc)
        closes = [100.0 + 0.10 * i for i in range(30)] + [103.0 - 0.14 * i for i in range(1, 16)]
        for i, c in enumerate(closes):
            state.bars.append(
                {
                    "t": (base + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "o": c,
                    "h": c + 0.05,
                    "l": c - 0.05,
                    "c": c,
                    "v": 1000.0 if i < 30 else 3000.0,
                }
            )
        state.last_price = closes[-1]
        result = _dispatch_tool("detect_regime_shift", {"symbol": "AAPL"}, app, DecisionTracker())
        assert result["shift_detected"] is True
        assert result["bias"] == "exit_or_stand_aside"
        assert result["levels_stale"] is True

    def test_detect_regime_shift_without_bars_returns_note(self):
        app, _ = _app()
        result = _dispatch_tool("detect_regime_shift", {"symbol": "AAPL"}, app, DecisionTracker())
        assert "note" in result

    def test_reversal_personality_uses_reversal_tools(self):
        assert PERSONALITY_TOOLS["reversal"] is REVERSAL_TOOLS
        names = {t["function"]["name"] for t in REVERSAL_TOOLS}
        assert {"analyze_vwap_bands", "vwap_reversion_geometry", "analyze_volume", "get_quote"} <= names
        assert "analyze_daily_trend" not in names
        assert "analyze_opening_range" not in names
        assert "get_put_call_walls" not in names

    def test_vwap_reversion_geometry_tool_computes_reward_risk(self):
        result = _tool_vwap_reversion_geometry(entry=98.0, vwap=100.0, std_dev=1.0, side="long")
        assert result["reward_risk_ratio"] == 2.0
        assert result["meets_min_reward_risk"] is True

    def test_analyze_vwap_bands_dispatches(self):
        app, _ = _app()
        result = _dispatch_tool("analyze_vwap_bands", {}, app, DecisionTracker())
        assert "note" in result  # no bars yet

    def test_dispatch_rejects_unknown_symbol(self):
        app, _ = _app()
        result = _dispatch_tool("analyze_vwap_bands", {"symbol": "ZZZ"}, app, DecisionTracker())
        assert "error" in result and "ZZZ" in result["error"]

    def test_dispatch_routes_symbol_to_its_state(self):
        app, _ = _app("AAPL", "TSLA")
        app.sym("TSLA").news = [{"id": "1", "headline": "tsla news", "summary": "", "created_at": "t", "source": "x"}]
        result = _dispatch_tool("get_news", {"symbol": "TSLA"}, app, DecisionTracker())
        assert result["articles"][0]["headline"] == "tsla news"
        result = _dispatch_tool("get_news", {"symbol": "AAPL"}, app, DecisionTracker())
        assert result["articles"] == []

    def test_smart_money_personality_uses_smart_money_tools(self):
        assert PERSONALITY_TOOLS["smart_money"] is SMART_MONEY_TOOLS
        names = {t["function"]["name"] for t in SMART_MONEY_TOOLS}
        assert {
            "analyze_daily_trend",
            "analyze_order_blocks",
            "analyze_smart_money_setup",
            "analyze_fair_value_gaps",
            "smart_money_trade_geometry",
        } <= names
        assert "analyze_market" not in names
        assert "get_put_call_walls" not in names
        assert "analyze_opening_range" not in names
        assert "analyze_vwap_bands" not in names

    def test_smart_money_trade_geometry_tool_computes_reward_risk(self):
        result = _tool_smart_money_trade_geometry(entry=100.0, stop=98.0, target=110.0)
        assert result["reward_risk_ratio"] == 5.0
        assert result["meets_min_reward_risk"] is True

    def test_analyze_smart_money_setup_dispatches_without_daily_bars(self):
        app, _ = _app()
        result = _dispatch_tool("analyze_smart_money_setup", {}, app, DecisionTracker())
        assert "note" in result  # no daily bars yet

    def test_analyze_order_blocks_dispatches_without_daily_bars(self):
        app, _ = _app()
        result = _dispatch_tool("analyze_order_blocks", {}, app, DecisionTracker())
        assert "note" in result  # no daily bars yet


class TestRunAgentCycle:
    def test_records_buy_decision_and_logs_tool_calls(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        state.feed = "iex"
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=100.0), trade_cost=0.0)

        responses = [
            _response(tool_calls=[_tool_call("c1", "analyze_daily_trend", {"limit": 60})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {"action": "buy", "quantity": 2, "regime": "bullish", "reasoning": "uptrend confirmed"},
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-3.5-flash", ["AAPL"], state, tracker, max_iters=5)

        snap = tracker.snapshot()
        assert snap["positions"] == {"AAPL": 2.0}
        assert snap["cash"] == 800.0
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "buy"

        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "tool_call" in log_types
        assert "decision" in log_types

    def test_breakout_personality_passes_breakout_tools_to_client(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "no setup yet, watch the opening-range high",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-3.5-flash", ["AAPL"], state, tracker, max_iters=3, personality="breakout")

        assert client.tools_seen[0] is BREAKOUT_TOOLS

    def test_forces_sleep_when_max_iters_reached_without_decision(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [_response(content="still thinking...") for _ in range(3)]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-3.5-flash", ["AAPL"], state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "sleep"

    def test_removed_sleep_action_is_rejected_and_retried(self):
        """The agent can no longer choose to sleep. A model that still reaches for the
        old "sleep" action gets an error back and must finalize with a real decision --
        here it corrects to an alert."""
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[_tool_call("c1", "submit_decision", {"action": "sleep", "reasoning": "no clear edge"})]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "stand aside until price reclaims resistance",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-3.5-flash", ["AAPL"], state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "alert"
        assert state.sym("AAPL").alerts == [{"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}]

        # the rejected "sleep" attempt must have been surfaced back to the model
        second_call_messages = client.calls[1]
        tool_results = [m["content"] for m in second_call_messages if m.get("role") == "tool"]
        assert any("'buy', 'sell', or 'alert'" in c for c in tool_results)

    def test_alert_decision_sets_state_alert_and_records_no_trade(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "wait for breakout above resistance",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-3.5-flash", ["AAPL"], state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        decision = snap["decisions"][0]
        assert decision.action == "alert"
        assert decision.status == "noop"
        assert decision.price is None
        assert decision.alerts == [{"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}]
        assert state.sym("AAPL").alerts == [{"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}]

    def test_alert_decision_supports_multiple_conditions_across_fields(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "watch a breakdown below support or a real volume surge",
                            "alerts": [
                                {"field": "last_price", "condition": "below", "value": 95.0},
                                {"field": "volume_ratio", "condition": "above", "value": 2.0},
                            ],
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gemini-3.5-flash", ["AAPL"], state, tracker, max_iters=3)

        snap = tracker.snapshot()
        decision = snap["decisions"][0]
        assert decision.action == "alert"
        assert decision.alerts == [
            {"symbol": "AAPL", "field": "last_price", "condition": "below", "value": 95.0},
            {"symbol": "AAPL", "field": "volume_ratio", "condition": "above", "value": 2.0},
        ]
        assert state.sym("AAPL").alerts == decision.alerts

    def test_alert_with_missing_fields_is_rejected_and_retried(self):
        """An incomplete alert call (no conditions) must not be silently downgraded to
        sleep -- the model gets an error back and a chance to correct itself, since it
        otherwise has no way to know its alert was dropped."""
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "submit_decision", {"action": "alert", "reasoning": "no level chosen"})
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "watching for a breakout above resistance",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gpt-4.1-mini", ["AAPL"], state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "alert"
        assert snap["decisions"][0].alerts == [{"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}]
        assert state.sym("AAPL").alerts == [{"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}]

        # the rejected first attempt must have been surfaced back to the model as a tool result
        first_call_messages = client.calls[1]
        tool_results = [m["content"] for m in first_call_messages if m.get("role") == "tool"]
        assert any("requires a non-empty 'alerts' array" in c for c in tool_results)

    def test_alert_with_invalid_field_is_rejected(self):
        """A condition naming a field that isn't continuously tracked is dropped, so an
        otherwise-empty alert is rejected rather than silently watching nothing. The model
        then corrects to a valid, watchable condition."""
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "bad field",
                            "alerts": [{"field": "rsi", "condition": "above", "value": 70}],
                        },
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "stand aside until a real breakdown",
                            "alerts": [{"field": "last_price", "condition": "below", "value": 95.0}],
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gpt-4.1-mini", ["AAPL"], state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert snap["decisions"][0].action == "alert"
        assert state.sym("AAPL").alerts == [{"symbol": "AAPL", "field": "last_price", "condition": "below", "value": 95.0}]

    def test_alert_with_missing_fields_falls_back_to_sleep_if_never_corrected(self):
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())

        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "submit_decision", {"action": "alert", "reasoning": "no level chosen"})
                ]
            )
            for _ in range(3)
        ]
        client = FakeClient(responses)

        run_agent_cycle(client, "gpt-4.1-mini", ["AAPL"], state, tracker, max_iters=3)

        snap = tracker.snapshot()
        assert len(snap["decisions"]) == 1
        assert snap["decisions"][0].action == "sleep"
        assert state.iter_alerts() == []


class TestSessionAwareness:
    # 2026-07-08 is a Wednesday; ET is UTC-4 in July.
    OPEN_UTC = datetime(2026, 7, 8, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, in session
    PRE_OPEN_UTC = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)  # 08:00 ET, pre-market

    def test_addendum_empty_while_market_open(self):
        assert _session_closed_addendum(self.OPEN_UTC) == ""

    def test_addendum_names_next_open_and_minutes_when_closed(self):
        text = _session_closed_addendum(self.PRE_OPEN_UTC)
        assert "MARKET CLOSED" in text
        assert "2026-07-08 09:30" in text
        assert "90 minutes" in text

    def _run_cycle(self, personality: str = "momentum") -> FakeClient:
        state = AppState()
        state.set_symbols(["AAPL"])
        state.api_key = "k"
        state.api_secret = "s"
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "waiting for the open",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 1.0}],
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)
        run_agent_cycle(client, "gpt-4.1-mini", ["AAPL"], state, tracker, max_iters=2, personality=personality)
        return client

    def test_cycle_prompt_carries_addendum_when_market_closed(self, monkeypatch):
        monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: False)
        client = self._run_cycle()
        system_prompt = client.calls[0][0]["content"]
        assert "MARKET CLOSED" in system_prompt

    def test_cycle_prompt_clean_while_market_open(self, monkeypatch):
        monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: True)
        client = self._run_cycle()
        system_prompt = client.calls[0][0]["content"]
        assert "MARKET CLOSED" not in system_prompt

    def test_premarket_personality_is_exempt(self, monkeypatch):
        monkeypatch.setattr(market_hours, "is_market_open", lambda now=None: False)
        client = self._run_cycle(personality="premarket")
        system_prompt = client.calls[0][0]["content"]
        assert "MARKET CLOSED" not in system_prompt


class TestAlertTrigger:
    def test_alert_triggered_above(self):
        import time
        _, state = _app()
        state.recent_prices.append((time.monotonic(), 151.0))
        assert alert_triggered(state, {"field": "last_price", "condition": "above", "value": 150.0}) is True
        state.recent_prices.clear()
        state.recent_prices.append((time.monotonic(), 149.0))
        assert alert_triggered(state, {"field": "last_price", "condition": "above", "value": 150.0}) is False

    def test_alert_triggered_below(self):
        import time
        _, state = _app()
        state.recent_prices.append((time.monotonic(), 99.0))
        assert alert_triggered(state, {"field": "last_price", "condition": "below", "value": 100.0}) is True
        state.recent_prices.clear()
        state.recent_prices.append((time.monotonic(), 101.0))
        assert alert_triggered(state, {"field": "last_price", "condition": "below", "value": 100.0}) is False

    def test_alert_triggered_on_non_price_field(self):
        _, state = _app()
        state.day_volume = 6_000_000
        assert alert_triggered(state, {"field": "day_volume", "condition": "above", "value": 5_000_000}) is True
        state.bid_price, state.ask_price = 10.0, 10.05
        assert alert_triggered(state, {"field": "spread", "condition": "above", "value": 0.04}) is True
        assert alert_triggered(state, {"field": "spread", "condition": "below", "value": 0.04}) is False

    def test_alert_triggered_on_momentum_pct(self):
        from datetime import datetime, timedelta, timezone
        _, state = _app()
        t0 = datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)
        # Price fell 2% over the 10-minute momentum window.
        state.bars.append({"t": t0.isoformat(), "c": 100.0})
        state.bars.append({"t": (t0 + timedelta(minutes=10)).isoformat(), "c": 98.0})
        assert alert_triggered(state, {"field": "momentum_pct", "condition": "below", "value": -1.0}) is True
        assert alert_triggered(state, {"field": "momentum_pct", "condition": "above", "value": 0.0}) is False

    def test_wait_returns_early_when_alert_fires(self):
        import time
        state, sym_state = _app()
        sym_state.last_price = 151.0
        sym_state.recent_prices.append((time.monotonic(), 151.0))
        sym_state.alerts = [{"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0}]
        stop_event = threading.Event()

        start = threading.Event()
        finished = threading.Event()

        def run():
            start.set()
            _wait_for_next_cycle(state, stop_event, cycle_sec=60)
            finished.set()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)

        assert finished.is_set()
        assert sym_state.alerts == []  # cleared once triggered
        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "status" in log_types

    def test_wait_returns_early_when_low_side_of_bracket_fires(self):
        import time
        state, sym_state = _app()
        sym_state.last_price = 94.0
        sym_state.recent_prices.append((time.monotonic(), 94.0))
        sym_state.alerts = [
            {"symbol": "AAPL", "field": "last_price", "condition": "below", "value": 95.0},
            {"symbol": "AAPL", "field": "last_price", "condition": "above", "value": 150.0},
        ]
        stop_event = threading.Event()

        start = threading.Event()
        finished = threading.Event()

        def run():
            start.set()
            _wait_for_next_cycle(state, stop_event, cycle_sec=60)
            finished.set()

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)

        assert finished.is_set()
        assert sym_state.alerts == []  # both levels cleared once either fires
        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "status" in log_types

    def test_wait_returns_early_when_woken_externally(self):
        """The actual news/alert detection now lives in the stream callbacks (see
        stream.py), which signal `agent_wake_event` directly. This simulates that
        external signal to verify `_wait_for_next_cycle` is a real, event-driven
        block rather than a self-polling loop."""
        state, _ = _app()
        stop_event = threading.Event()

        def signal_after_start():
            start.wait()
            state.agent_wake_reason = "Fresh news arrived for the ticker."
            state.agent_wake_event.set()

        start = threading.Event()
        finished = threading.Event()

        def run():
            start.set()
            _wait_for_next_cycle(state, stop_event, cycle_sec=60)
            finished.set()

        signaler = threading.Thread(target=signal_after_start)
        thread = threading.Thread(target=run)
        thread.start()
        signaler.start()
        thread.join(timeout=5)
        signaler.join(timeout=5)

        assert finished.is_set()
        with state.lock:
            log_types = [e["type"] for e in state.agent_log]
        assert "status" in log_types

    def test_wait_respects_stop_event_without_alert(self):
        state = AppState()
        stop_event = threading.Event()
        stop_event.set()  # already stopped, should return immediately

        _wait_for_next_cycle(state, stop_event, cycle_sec=60)  # should not hang


class TestOpeningRangeTool:
    """_tool_analyze_opening_range: cache-first, honest refusal, no fabrication."""

    @staticmethod
    def _bar(ts, h, l, c, v=1000):
        return {"t": ts, "o": c, "h": h, "l": l, "c": c, "v": v}

    def _open_bars(self):
        # 2026-07-16 is EDT: 09:30 ET = 13:30Z. 15 bars spanning 100-102.
        return [
            self._bar(f"2026-07-16T13:{30 + i}:00Z", 102.0 if i == 5 else 101.0, 100.0, 100.5)
            for i in range(15)
        ]

    def test_measured_range_is_cached_on_state(self, monkeypatch):
        from datetime import datetime as real_datetime

        from agent_stonks.agent import _tool_analyze_opening_range

        _, state = _app()
        state.bars.extend(self._open_bars())
        # Pin "today" to the bars' session so the cache date check matches.
        class _FakeDatetime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return real_datetime(2026, 7, 16, 14, 0, tzinfo=tz)

        import agent_stonks.agent as agent_mod

        monkeypatch.setattr(agent_mod, "datetime", _FakeDatetime)
        result = _tool_analyze_opening_range(state, minutes=15)
        assert result["opening_range_high"] == 102.0
        assert state.opening_range is not None
        assert state.opening_range["date"] == "2026-07-16"
        assert state.opening_range["complete"] is True

    def test_mid_session_buffer_returns_note_without_keys(self):
        from agent_stonks.agent import _tool_analyze_opening_range

        _, state = _app()  # no API keys -> no REST recovery
        state.bars.extend(
            [self._bar("2026-07-16T16:45:00Z", 105.0, 104.0, 104.5),
             self._bar("2026-07-16T16:46:00Z", 105.5, 104.5, 105.0)]
        )
        result = _tool_analyze_opening_range(state, minutes=15)
        assert "opening_range_high" not in result
        assert "note" in result


class TestParticipationUnification:
    def _bars(self):
        return [
            {"t": f"2026-07-21T14:{m:02d}:00Z", "o": 100.0, "h": 101.0, "l": 99.0,
             "c": 100.5, "v": 10_000.0 + m}
            for m in range(30)
        ]

    def test_analyze_volume_reports_armable_pace(self, monkeypatch):
        # The executor evaluates armed rvol_pace conditions from the trading
        # feed's counters, not the consolidated tape -- the tool must surface
        # that exact value so the agent arms thresholds against it.
        import agent_stonks.agent as agent_module

        _, state = _app()
        monkeypatch.setattr(agent_historical, "fetch_intraday_volume_bars", lambda symbol: self._bars())
        monkeypatch.setattr(agent_historical, "fetch_daily_volume_bars", lambda symbol: [])
        monkeypatch.setattr(agent_module, "rvol_pace", lambda day_volume, daily_bars: 1.234)
        result = _tool_analyze_volume(state)
        assert result["rvol_pace_armable"] == 1.23
        assert "rvol_pace" in result
        assert "Armed tactic conditions" in result["summary"]

    def test_armable_pace_none_when_unavailable(self, monkeypatch):
        import agent_stonks.agent as agent_module

        _, state = _app()
        monkeypatch.setattr(agent_historical, "fetch_intraday_volume_bars", lambda symbol: self._bars())
        monkeypatch.setattr(agent_historical, "fetch_daily_volume_bars", lambda symbol: [])
        monkeypatch.setattr(agent_module, "rvol_pace", lambda day_volume, daily_bars: None)
        result = _tool_analyze_volume(state)
        assert result["rvol_pace_armable"] is None
        assert "Armed tactic conditions" not in (result.get("summary") or "")
