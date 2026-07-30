"""The SimLab simulation engine.

Replays a trading agent against stored minute bars, reusing the *real* agent
machinery -- ``run_agent_cycle``, ``DecisionTracker``, ``TacticsExecutor``,
the tool handlers, and the validation/logging around them -- so a strategy
tested here is exactly the strategy that trades live. Three substitutions make
that possible:

- the clock is pinned to the simulated moment (``agent_stonks.clock``), so
  every "what time / date is it" read inside a cycle lands on the tape;
- ``SimBroker`` fills trades at the stored tape price at the simulated moment
  (the agent still never picks its own fill price);
- live fetches are rerouted to the dataset (``simlab.patches``).

Between agent cycles the engine fast-forwards deterministically, bar by bar:
armed tactics are evaluated by the real ``TacticsExecutor`` matching logic
(called synchronously instead of from a thread), condition alerts by the real
``alert_triggered``, and stored news timestamps produce the same
wake-the-agent interrupt the live stream would. Hours of waiting collapse
into however many LLM cycles the session actually needed.

The rule-based Apple Trader (``agent_stonks.apple_trader``) replays through
the same engine on a different day loop: it has no prompt, no tools and no
LLM, and it scores *every* closed bar of the session instead of sleeping on
armed conditions, so there is nothing to fast-forward past -- see
``_run_rule_day``.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

from agent_stonks import clock, market_hours, persistence_model
from agent_stonks.agent import (
    AGENT_PERSONALITIES,
    DEFAULT_PERSONALITY,
    MOMENTUM_TOOLS,
    PERSONALITY_TOOLS,
    PREMARKET_PERSONALITY,
    run_agent_cycle,
)
from agent_stonks.apple_trader import APPLE_TRADER_KEY, AppleTrader, AppleTraderConfig
from agent_stonks.apple_trader import TICKER as APPLE_TRADER_TICKER
from agent_stonks.broker import Broker
from agent_stonks.config import PREMARKET_LEAD_SEC
from agent_stonks.decisions import DecisionTracker
from agent_stonks.market_hours import MARKET_TZ
from agent_stonks.state import AppState, alert_triggered, format_alert
from agent_stonks.tactics import TacticsExecutor

from .market import BAR_SEC, SimMarket
from .patches import simulation_context

logger = logging.getLogger(__name__)

# Generous bar buffer: a full stored day (04:00-20:00 ET) is 960 minutes, and
# session-anchored reads (opening range, VWAP) must never lose the 09:30 open.
SIM_MAX_BARS = 1200

ProgressCb = Callable[[str], None]


@dataclass
class SimulationConfig:
    personality: str
    provider: str
    model: str
    api_key: str
    symbols: list[str]
    days: list[date]
    starting_cash: float = 100_000.0
    # Re-cycle cadence while the agent has nothing armed (live default is 60s;
    # simulation defaults coarser to keep LLM cost per session sane).
    cycle_minutes: int = 5
    # Hard cap on LLM cycles per simulated day -- a runaway agent (e.g. alerts
    # armed a hair from the price) stops burning tokens and fast-forwards.
    max_cycles_per_day: int = 40
    system_prompt_override: Optional[str] = None
    # Apple Trader's rule set, as a JSON-ready dict (so it survives the
    # experiment record). Ignored by every LLM personality; `provider`,
    # `model`, `api_key`, `cycle_minutes` and `max_cycles_per_day` are ignored
    # in return, since the rule agent scores every bar and calls no LLM.
    rule_config: Optional[dict] = None


@dataclass
class SimulationResult:
    config_summary: dict
    decisions: list[dict]
    agent_log: list[dict]
    equity: list[dict]  # [{ts, value}]
    cycles_run: int
    final_value: float
    starting_cash: float
    error: Optional[str] = None
    interrupted: bool = False
    prompt_used: Optional[str] = None
    tool_names: list[str] = field(default_factory=list)


class SimBroker(Broker):
    """Fills at the stored tape price at the pinned simulated moment."""

    def __init__(self, market: SimMarket) -> None:
        self.market = market

    def get_current_price(self, symbol: str, key: str, secret: str, feed: str = "iex") -> float:
        price = self.market.price_at(symbol.upper(), clock.now())
        if price is None:
            raise RuntimeError(f"no simulated tape price for {symbol} at {clock.now().isoformat()}")
        return price

    def submit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        return {"status": "filled", "filled_qty": quantity, "filled_price": price}


class SimulationEngine:
    def __init__(self, market: SimMarket, config: SimulationConfig, progress: ProgressCb = lambda m: None) -> None:
        self.market = market
        self.config = config
        self.progress = progress
        self.app = AppState()
        self.app.set_symbols(config.symbols)
        # Non-empty placeholder credentials: tools gate on their presence, and
        # every data path that would use them is dataset-backed in simulation.
        self.app.api_key = "simulated"
        self.app.api_secret = "simulated"
        self.tracker = DecisionTracker(starting_cash=config.starting_cash, broker=SimBroker(market))
        self.app.decision_tracker = self.tracker
        self.app.starting_budget = config.starting_cash
        self.executors: dict[str, TacticsExecutor] = {}
        for ss in self.app.iter_symbol_states():
            ss.bars = deque(maxlen=SIM_MAX_BARS)
            # Evaluated synchronously per simulated bar -- never .start()ed.
            self.executors[ss.symbol] = TacticsExecutor(ss, self.tracker)
        self.equity: list[dict] = []
        self.cycles_run = 0
        self._cursors: dict[str, int] = {sym: 0 for sym in self.app.symbols}
        self._last_step: Optional[datetime] = None
        self._stop_requested = False
        # The momentum-persistence bundle, loaded once per rule-based run.
        self._bundle: Optional[dict] = None

    @property
    def rule_based(self) -> bool:
        """Whether this run is the non-LLM Apple Trader loop."""
        return self.config.personality == APPLE_TRADER_KEY

    # ------------------------------------------------------------------ API

    def request_stop(self) -> None:
        """Ask a running simulation to stop after the current step/cycle."""
        self._stop_requested = True

    def seed_until(self, t: datetime) -> None:
        """Advance the tape to `t` without running any agent cycles -- used to
        probe tool outputs at an arbitrary stored moment (the Agents tab's
        hand-testing). Call inside `simulation_context`."""
        for day in self.market.days:
            for step in self.market.step_times(day):
                if step <= t:
                    self._apply_step(step)

    def run(self, client: Any = None) -> SimulationResult:
        """Run the whole simulation. `client` overrides the LLM client (tests
        inject a fake); by default one is built from the config. The rule-based
        agent needs no client at all."""
        if client is None and not self.rule_based:
            from agent_stonks.llm import get_agent_client

            client = get_agent_client(self.config.provider, self.config.api_key)

        error: Optional[str] = None
        with simulation_context(self.market):
            try:
                run_day = self._run_rule_day if self.rule_based else self._run_day
                trader = self._build_rule_trader() if self.rule_based else client
                for day in self.market.days:
                    if self._stop_requested:
                        break
                    run_day(day, trader)
            except Exception as exc:  # surface, don't lose the partial run
                logger.exception("simulation failed")
                error = str(exc)

        final_value = self.equity[-1]["value"] if self.equity else self.config.starting_cash
        snap = self.tracker.snapshot()
        return SimulationResult(
            config_summary={
                "personality": self.config.personality,
                "provider": self.config.provider,
                "model": self.config.model,
                "symbols": self.app.symbols,
                "days": [d.isoformat() for d in self.market.days],
                "starting_cash": self.config.starting_cash,
                "cycle_minutes": self.config.cycle_minutes,
                "prompt_overridden": self.config.system_prompt_override is not None,
                "rule_based": self.rule_based,
                "rule_config": self.config.rule_config if self.rule_based else None,
            },
            decisions=[asdict(d) for d in snap["decisions"]],
            agent_log=list(self.app.agent_log),
            equity=self.equity,
            cycles_run=self.cycles_run,
            final_value=final_value,
            starting_cash=self.config.starting_cash,
            error=error,
            interrupted=self._stop_requested,
            # Resolved exactly as run_agent_cycle resolves them, so a stored
            # run says which prompt and which tools actually produced it --
            # `prompt_overridden` alone can't tell two built-in prompt
            # revisions apart when comparing runs weeks later.
            prompt_used=self._resolved_prompt(),
            tool_names=self._resolved_tool_names(),
        )

    def _resolved_prompt(self) -> Optional[str]:
        # The rule agent has no prompt to record: its rules ARE its identity,
        # and they are stored under `rule_config` instead.
        if self.rule_based:
            return None
        return self.config.system_prompt_override or AGENT_PERSONALITIES.get(
            self.config.personality, AGENT_PERSONALITIES[DEFAULT_PERSONALITY]
        )["system_prompt"]

    def _resolved_tool_names(self) -> list[str]:
        if self.rule_based:
            return []
        tools = PERSONALITY_TOOLS.get(self.config.personality, MOMENTUM_TOOLS)
        return [t["function"]["name"] for t in tools]

    # ------------------------------------------------------------- stepping

    def _apply_step(self, t: datetime) -> None:
        """Pin the clock to `t` and fold every bar completed by `t` into the
        per-symbol states (price, previous-minute fields, day volume,
        recent-price window, daily bars)."""
        clock.set_simulated(t)
        for ss in self.app.iter_symbol_states():
            series = self.market.series[ss.symbol]
            cursor = self._cursors[ss.symbol]
            fresh: list[dict] = []
            while cursor < len(series.minute_ts) and series.minute_ts[cursor] + timedelta(seconds=BAR_SEC) <= t:
                fresh.append(series.minute_bars[cursor])
                cursor += 1
            self._cursors[ss.symbol] = cursor
            if not fresh and ss.last_price is not None:
                continue
            with ss.lock:
                for bar in fresh:
                    ss.bars.append(bar)
                if fresh:
                    last = fresh[-1]
                    close = float(last["c"])
                    ss.last_price = close
                    ss.previous_minute_high = float(last["h"])
                    ss.previous_minute_low = float(last["l"])
                    ss.previous_minute_close = close
                    ss.quote_ts = t.isoformat()
                    mono = clock.monotonic()
                    for key in ("o", "h", "l", "c"):
                        try:
                            ss.recent_prices.append((mono, float(last[key])))
                        except (KeyError, TypeError, ValueError):
                            continue
                ss.day_volume = self._day_volume(ss.symbol, t)
            ss.daily_bars = self.market.daily_bars_at(ss.symbol, t)
            prev = self.market.prev_close(ss.symbol, t)
            if prev is not None:
                ss.prev_close = prev
            ss.news = self.market.news_at(ss.symbol, t)
        value = self.app.mark_to_market()
        if value is not None:
            self.equity.append({"ts": t.isoformat(), "value": value})
        self._last_step = t

    def _day_volume(self, symbol: str, t: datetime) -> float:
        series = self.market.series[symbol]
        today = t.astimezone(MARKET_TZ).date()
        total = 0.0
        for bar, ts in zip(series.minute_bars, series.minute_ts):
            if ts + timedelta(seconds=BAR_SEC) > t:
                break
            if ts.astimezone(MARKET_TZ).date() == today:
                total += float(bar.get("v") or 0.0)
        return total

    def _check_wake(self, prev_t: datetime, t: datetime) -> Optional[str]:
        """Evaluate tactics, alerts, and news at step `t`. Returns the wake
        reason (and consumes the wake state) or None to keep fast-forwarding."""
        # Tactics: the real matching engine, run synchronously. An execution
        # sets the app wake event exactly as it does live.
        for executor in self.executors.values():
            executor.check_now()
        if self.app.agent_wake_event.is_set():
            reason = self.app.agent_wake_reason or "Tactics executed."
            self.app.agent_wake_event.clear()
            self.app.agent_wake_reason = None
            return reason

        for ss, alert in self.app.iter_alerts():
            if alert_triggered(ss, alert):
                self.app.clear_alerts()
                return f"Alert triggered: {format_alert(alert)}."

        for ss in self.app.iter_symbol_states():
            fresh = self.market.fresh_news(ss.symbol, prev_t, t)
            if fresh:
                headline = str(fresh[0].get("headline") or "")[:120]
                return f"Fresh news for {ss.symbol}: {headline}"
        return None

    def _log(self, entry_type: str, text: str) -> None:
        with self.app.lock:
            self.app.agent_log.append({"ts": clock.now().isoformat(), "type": entry_type, "text": text})

    # ------------------------------------------------------------ day loops

    def _first_cycle_time(self, day: date, steps: list[datetime]) -> Optional[datetime]:
        if self.config.personality == PREMARKET_PERSONALITY:
            target = self.market.session_open(day) - timedelta(seconds=PREMARKET_LEAD_SEC)
        else:
            target = self.market.session_open(day) + timedelta(seconds=BAR_SEC)
        return next((t for t in steps if t >= target), None)

    # --------------------------------------------------- rule-based day loop

    def _build_rule_trader(self) -> AppleTrader:
        """The Apple Trader state machine plus its model, or a clear failure.

        Both problems this catches are configuration mistakes that would
        otherwise show up as an empty run: the ticker the model is fitted for
        not being in the dataset, and the saved bundle not being installed.
        """
        if APPLE_TRADER_TICKER not in self.app.symbols:
            raise RuntimeError(
                f"Apple Trader only trades {APPLE_TRADER_TICKER}; this simulation's "
                f"symbols are {', '.join(self.app.symbols) or 'none'}."
            )
        self._bundle = persistence_model.load_bundle()
        if self._bundle is None:
            raise RuntimeError(
                f"no momentum-persistence model at {persistence_model.model_path()} "
                "(or scikit-learn/joblib are not installed)."
            )
        config = AppleTraderConfig(**(self.config.rule_config or {}))
        # Every indicator the model consumes is session-local, so one stored
        # day is a complete, like-live run -- there is nothing behind it to
        # miss.
        return AppleTrader(
            config, model_threshold=persistence_model.model_threshold(self._bundle)
        )

    def _run_rule_day(self, day: date, trader: AppleTrader) -> None:
        """Score every closed bar of the session, in order.

        No wake logic and no fast-forward: the rule agent arms no tactics and
        no alerts, and its whole design is one decision per closed minute, so
        the day is simply walked. Bars outside the regular session are still
        applied (equity keeps marking) but never scored -- live the loop is
        idle then too.
        """
        for t in self.market.step_times(day):
            if self._stop_requested:
                return
            self._apply_step(t)
            if not market_hours.is_market_open(t):
                continue
            self.cycles_run += 1
            outcome = trader.run_cycle(self._bundle, self.app, self.tracker)
            if outcome in ("bought", "sold") or self.cycles_run % 60 == 0:
                self.progress(
                    f"{day.isoformat()} {t.astimezone(MARKET_TZ).strftime('%H:%M')} — "
                    f"bar {self.cycles_run} ({outcome})"
                )

    # --------------------------------------------------------- LLM day loop

    def _run_day(self, day: date, client: Any) -> None:
        steps = self.market.step_times(day)
        if not steps:
            return
        first_cycle = self._first_cycle_time(day, steps)
        if first_cycle is None:
            return
        close = self.market.session_close(day)

        # Seed the tape up to the first cycle without wake checks.
        idx = 0
        while idx < len(steps) and steps[idx] <= first_cycle:
            self._apply_step(steps[idx])
            idx += 1

        premarket = self.config.personality == PREMARKET_PERSONALITY
        premarket_retired = False
        cycles_today = 0
        run_cycle_now = True
        last_cycle_at = first_cycle

        while idx <= len(steps):
            if self._stop_requested:
                return
            if run_cycle_now and not premarket_retired:
                if cycles_today >= self.config.max_cycles_per_day:
                    self._log(
                        "status",
                        f"Reached max cycles for {day.isoformat()} "
                        f"({self.config.max_cycles_per_day}); fast-forwarding to the close.",
                    )
                    premarket_retired = True  # no more cycles today; tactics stay live
                else:
                    cycles_today += 1
                    self.cycles_run += 1
                    self.progress(
                        f"{day.isoformat()} {clock.now().astimezone().strftime('%H:%M')} — "
                        f"cycle {cycles_today}"
                    )
                    self.app.agent_wake_event.clear()
                    self.app.agent_wake_reason = None
                    run_agent_cycle(
                        client,
                        self.config.model,
                        self.app.symbols,
                        self.app,
                        self.tracker,
                        personality=self.config.personality,
                        system_prompt_override=self.config.system_prompt_override,
                    )
                    last_cycle_at = clock.now()
                    if premarket:
                        # One-shot pre-open specialist: no re-cycling once the
                        # session is live; a pre-bell news wake revises the plan.
                        premarket_retired = clock.now() >= self.market.session_open(day)
                run_cycle_now = False

            if idx >= len(steps):
                break
            t = steps[idx]
            idx += 1
            prev_t = self._last_step or t
            self._apply_step(t)
            if t > close:
                # After-hours tape: keep marking equity, stop trading cycles.
                continue

            reason = self._check_wake(prev_t, t)
            if reason is not None:
                if premarket:
                    executed = reason.startswith("Tactics executed")
                    pre_bell = t < self.market.session_open(day)
                    if executed:
                        self._log("status", f"{reason} Premarket analyst retiring.")
                        premarket_retired = True
                    elif pre_bell and reason.startswith("Fresh news"):
                        self._log("status", f"{reason} Revising the opening plan.")
                        run_cycle_now = True
                    continue
                self._log("status", f"{reason} Waking early.")
                run_cycle_now = True
                continue

            # No alerts and no tactics armed: the plain cycle timer applies.
            if (
                not premarket
                and not premarket_retired
                and not self.app.any_tactics()
                and not self.app.iter_alerts()
                and (t - last_cycle_at) >= timedelta(minutes=self.config.cycle_minutes)
            ):
                run_cycle_now = True
