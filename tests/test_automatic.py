import json
from types import SimpleNamespace

from agent_stonks.agent import (
    DISABLED_PERSONALITIES,
    _TOOL_STAND_DOWN,
    run_agent_cycle,
)
from agent_stonks.automatic import (
    AUTOMATIC_KEY,
    REGIME_TOOLS,
    SELECTABLE_STRATEGIES,
    run_regime_cycle,
)
from agent_stonks.broker import Broker
from agent_stonks.decisions import DecisionTracker
from agent_stonks.state import AppState


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


def _base_state() -> AppState:
    state = AppState()
    state.set_symbols(["AAPL"])
    state.api_key = "k"
    state.api_secret = "s"
    state.feed = "iex"
    return state


class TestStandDown:
    def test_stand_down_tool_only_added_under_automatic(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        # Normal mode: model finalizes with a regular alert.
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "submit_decision",
                        {
                            "action": "alert",
                            "reasoning": "waiting",
                            "alerts": [{"field": "last_price", "condition": "above", "value": 150.0}],
                        },
                    )
                ]
            )
        ]
        client = FakeClient(responses)
        run_agent_cycle(client, "m", ["AAPL"], state, tracker, max_iters=3, personality="breakout")
        names = {t["function"]["name"] for t in client.tools_seen[0]}
        assert "stand_down" not in names

    def test_stand_down_added_and_returns_signal(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "stand_down",
                        {"reasoning": "range has resolved into a strong trend", "expected_quiet_minutes": 45},
                    )
                ]
            )
        ]
        client = FakeClient(responses)
        result = run_agent_cycle(
            client, "m", "AAPL", state, tracker, max_iters=3, personality="reversal", under_automatic=True
        )
        assert result == "stand_down"
        # stand_down tool was exposed to the model
        names = {t["function"]["name"] for t in client.tools_seen[0]}
        assert "stand_down" in names
        # No trade/alert decision was recorded -- it's a relinquish, not a decision.
        snap = tracker.snapshot()
        assert snap["decisions"] == []
        # It was logged as a stand_down event.
        with state.lock:
            types = [e["type"] for e in state.agent_log]
        assert "stand_down" in types

    def test_normal_decision_returns_decided(self):
        state = _base_state()
        tracker = DecisionTracker(starting_cash=1000.0, broker=FakeBroker(price=100.0), trade_cost=0.0)
        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "submit_decision", {"action": "buy", "quantity": 1, "reasoning": "go"})
                ]
            )
        ]
        client = FakeClient(responses)
        result = run_agent_cycle(
            client, "m", "AAPL", state, tracker, max_iters=3, personality="momentum", under_automatic=True
        )
        assert result == "decided"
        assert tracker.snapshot()["positions"] == {"AAPL": 1.0}


class TestRunRegimeCycle:
    def test_selects_strategy_after_analysis(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[_tool_call("c1", "analyze_daily_trend", {})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "c2",
                        "select_strategy",
                        {
                            "strategy": "momentum",
                            "regime": "bullish_trend",
                            "reasoning": "fresh gap on 3x volume with a catalyst",
                        },
                    )
                ]
            ),
        ]
        client = FakeClient(responses)
        selection = run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=5)
        assert selection["strategy"] == "momentum"
        assert selection["regime"] == "bullish_trend"
        assert client.tools_seen[0] is REGIME_TOOLS
        with state.lock:
            types = [e["type"] for e in state.agent_log]
        assert "regime_select" in types

    def test_invalid_strategy_is_rejected_then_corrected(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(tool_calls=[_tool_call("c1", "select_strategy", {"strategy": "scalping", "reasoning": "x"})]),
            _response(
                tool_calls=[
                    _tool_call("c2", "select_strategy", {"strategy": "momentum", "regime": "ranging", "reasoning": "mixed"})
                ]
            ),
        ]
        client = FakeClient(responses)
        selection = run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=5)
        assert selection["strategy"] == "momentum"
        # the rejected attempt was surfaced back to the model
        second_call = client.calls[1]
        tool_results = [m["content"] for m in second_call if m.get("role") == "tool"]
        assert any("strategy must be one of" in c for c in tool_results)

    def test_returns_none_when_never_selects(self):
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [_response(content="thinking") for _ in range(3)]
        client = FakeClient(responses)
        selection = run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=3)
        assert selection is None

    def test_regime_tools_include_corporate_actions_but_no_trading(self):
        names = {t["function"]["name"] for t in REGIME_TOOLS}
        assert "get_corporate_actions" in names
        assert "submit_decision" not in names
        assert "set_tactics" not in names

    def test_selectable_strategies_match_personalities(self):
        # Orchestrator can pick any tradeable personality, and automatic is not
        # itself selectable.
        assert AUTOMATIC_KEY not in SELECTABLE_STRATEGIES
        assert "momentum" in SELECTABLE_STRATEGIES
        assert "breakout" in SELECTABLE_STRATEGIES
        # Premarket is activated deterministically before the open, never by
        # the regime cycle.
        assert "premarket" not in SELECTABLE_STRATEGIES
        # Switched-off personalities are never activated either.
        assert not DISABLED_PERSONALITIES & set(SELECTABLE_STRATEGIES)
        enum = _TOOL_STAND_DOWN["function"]["parameters"]["properties"]["reasoning"]
        assert enum["type"] == "string"


class TestBreakoutActivationGate:
    """select_strategy('breakout') is deterministically rejected when the ORB
    strategy has nothing real to trade: an unfavorable session window, or no
    measurable opening range for today."""

    def test_gated_breakout_pick_is_rejected_then_corrected(self, monkeypatch):
        import agent_stonks.automatic as automatic_mod

        monkeypatch.setattr(
            automatic_mod,
            "breakout_preconditions",
            lambda app, symbols, minutes=15: "breakout is not selectable right now: midday dead zone",
        )
        state = _base_state()
        tracker = DecisionTracker(broker=FakeBroker())
        responses = [
            _response(
                tool_calls=[
                    _tool_call("c1", "select_strategy", {"strategy": "breakout", "regime": "breakout_pending", "reasoning": "x"})
                ]
            ),
            _response(
                tool_calls=[
                    _tool_call("c2", "select_strategy", {"strategy": "reversal", "regime": "ranging", "reasoning": "adx 15"})
                ]
            ),
        ]
        client = FakeClient(responses)
        selection = run_regime_cycle(client, "m", ["AAPL"], state, tracker, max_iters=5)
        assert selection["strategy"] == "reversal"
        tool_results = [m["content"] for m in client.calls[1] if m.get("role") == "tool"]
        assert any("not selectable" in c for c in tool_results)

    def test_preconditions_block_unfavorable_window(self, monkeypatch):
        import agent_stonks.technical_analysis as ta_mod
        from agent_stonks.agent import breakout_preconditions

        monkeypatch.setattr(
            ta_mod,
            "session_time_window",
            lambda *a, **k: {"favorable_for_breakouts": False, "summary": "12:30 ET -- midday dead zone."},
        )
        state = _base_state()
        reason = breakout_preconditions(state, ["AAPL"])
        assert reason is not None and "dead zone" in reason

    def test_preconditions_block_missing_opening_range(self, monkeypatch):
        import agent_stonks.technical_analysis as ta_mod
        from agent_stonks.agent import breakout_preconditions

        monkeypatch.setattr(
            ta_mod,
            "session_time_window",
            lambda *a, **k: {"favorable_for_breakouts": True, "summary": "10:00 ET -- opening window."},
        )
        state = AppState()
        state.set_symbols(["AAPL"])  # no API keys -> no REST recovery, no bars
        reason = breakout_preconditions(state, ["AAPL"])
        assert reason is not None and "opening range" in reason

    def test_preconditions_pass_with_cached_range(self, monkeypatch):
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        import agent_stonks.technical_analysis as ta_mod
        from agent_stonks.agent import breakout_preconditions

        monkeypatch.setattr(
            ta_mod,
            "session_time_window",
            lambda *a, **k: {"favorable_for_breakouts": True, "summary": "10:00 ET -- opening window."},
        )
        state = AppState()
        state.set_symbols(["AAPL"])
        today = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
        state.sym("AAPL").opening_range = {
            "date": today, "minutes": 15, "high": 102.0, "low": 100.0,
            "bar_count": 15, "avg_volume": 1000.0, "complete": True,
        }
        assert breakout_preconditions(state, ["AAPL"]) is None
