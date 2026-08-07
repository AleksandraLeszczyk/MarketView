"""Apple Trader -- a rule-based agent with no LLM in the loop.

Every other personality in `agent_stonks.agent` is a system prompt handed to a
model that reasons its way to a decision. This one is a plain loop: once a
minute it looks at the AAPL bar that just closed, and when the momentum regime
has just turned positive it asks a saved model from FinNotebooks/TimeToChange2
whether that change is the kind that holds. Same paper ledger, same fill path,
same log -- only the decision-making is deterministic, so the same tape always
produces the same trades.

Which model answers is a setting (`AppleTraderConfig.model_key`, see
`agent_stonks.apple_models`): the incumbent persistence classifier, or the
N-BEATS ensemble that forecasts momentum and derives the same probability from
500 sampled futures. The rules below do not change with it -- both are handed
the same 20-bar window on the same bars and return one number -- so a run's
model is part of its configuration identity rather than a different agent.

The rules
---------
* BUY when the last closed bar is a momentum-regime change **into positive**
  and the model's persistence probability for it is at least `prob_threshold`.
  That is the entire entry condition: one bar, one question, one answer. There
  is no confirmation count, because the thing a confirmation window would wait
  for -- the new regime surviving -- is exactly what the model is being asked.
  The one thing that overrides a signal is the clock: no position is opened
  inside the closing flatten window, because a long the next rule is about to
  shut is not a trade, it is two commissions.
* SELL when price has fallen `trail_pct` below the highest price seen since the
  entry. The peak ratchets up and never down, so the rule is a trailing stop
  that starts `trail_pct` under the entry and turns into a profit lock as the
  move runs.

Nothing else closes the position except the closing bell: momentum, regimes and
every feature the model uses are intraday and do not survive the overnight gap,
so the book is flattened `flatten_before_close_min` before the close.

Against the notebook
--------------------
The entry above is the rule TimeToChange2's own simulator runs
(`mshift.backtest.simulate_day`): the same to-positive-change trigger, the same
`proba >= threshold` test on the same 20-bar window, and the same refusal to
open a position it is about to be forced out of (there, "no decision on the
last bar of the session"). `tests/test_persistence_model.py` pins the feature
pipeline to `mshift` bar for bar, so a signal here is a signal there.

Two differences remain, deliberately:

* the **exit** is a trailing stop, where the notebook sells when actual
  momentum drops below a level. Different rule, different holding times -- and
  because a held position blocks the next entry, that alone can change which
  later signals become trades.
* the **fill**. The notebook decides at the close of bar *t* and fills at the
  open of bar *t+1*; live there is no such price to wait for, so this loop
  sends a market order as soon as the bar closes.

And one that is not in the code at all: the **tape**. The notebook's bars are
consolidated (yfinance); live, and in SimLab unless the dataset says otherwise,
they are Alpaca's IEX feed -- one venue, ~4% of consolidated volume. A few
cents of difference is enough to trip the regime trigger a minute earlier or
later, or to insert a regime change the other tape never saw, which moves
`pre_dwell` -- the model's strongest feature -- and with it the probability.

That difference is large enough to change the day's trades, so it is worth
being concrete about. On 2026-07-27 the consolidated tape has four to-positive
changes at 10:20, 11:00, 14:54 and 15:37 with `pre_dwell` 50, 28, 8 and 30. On
IEX the first slips to 10:21 and the 11:00 change comes in at `pre_dwell` 10
instead of 28 -- below the 15-bar precondition -- so that entry disappears and
the session trades twice instead of three times. Download the SimLab dataset
with `feed="sip"` and all four changes match the notebook minute for minute
and dwell for dwell; `simlab.data` keeps the two tapes as separate stores for
exactly this reason. A comparison against the notebook that does not check
which feed the dataset holds is not a comparison of the rules.

What the model actually does
----------------------------
TimeToChange2's own verdict on the incumbent classifier is that it is **a
filter that separates the impossible from the possible, not the likely from the
unlikely**: 0.82 out-of-fold AUC over all regime changes, but 0.50 over the
changes that already pass the observable "the old regime had held 15 bars"
pre-condition. The N-BEATS option is the one model in its benchmark that beats
chance on that hard half (0.67 +/- 0.07 over four folds), which is a real
effect and a small one -- see `apple_models` for what choosing it does and does
not buy.

Either way the entry is best read as "a to-positive change the model did not
veto", and the exit does not lean on the model at all -- once the position is
on, only price decides when it comes off.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

from . import apple_models, market_hours, persistence_model, scoring
from . import observability as obs
from .agent import _log, stop_agent
from .clock import now as _now
from .config import (
    APPLE_TRADER_BAR_LAG_SEC,
    APPLE_TRADER_CYCLE_SEC,
    APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN,
    APPLE_TRADER_MODEL,
    APPLE_TRADER_POSITION_PCT,
    APPLE_TRADER_PROB_THRESHOLD,
    APPLE_TRADER_TRAIL_PCT,
)
from .decisions import DecisionTracker
from .state import AppState

APPLE_TRADER_KEY = "apple_trader"
APPLE_TRADER_LABEL = "Apple Trader (rule-based, no LLM)"
APPLE_TRADER_AVATAR = "Multiavatar-4bcbffe68af819e050.png"

# Stands where a provider name goes for the other agents (SimLab run records,
# result grouping): this one has no LLM behind it, its rules are the "model".
RULE_PROVIDER = "rules"

# TimeToChange2 fitted a single AAPL model, and its own results do not claim to
# transfer -- the personality is named after the one symbol it can trade.
TICKER = "AAPL"


@dataclass
class AppleTraderConfig:
    """Tunables of the loop. `model_key`, `prob_threshold` and `trail_pct` are
    the three that change what it does; the rest are sizing and housekeeping."""

    # Which saved model answers the entry question -- a key of
    # `apple_models.MODELS`.
    model_key: str = APPLE_TRADER_MODEL
    # None -> the cut-off the chosen model picked on its own validation block,
    # which is the only setting that means the same thing across models.
    prob_threshold: Optional[float] = APPLE_TRADER_PROB_THRESHOLD
    trail_pct: float = APPLE_TRADER_TRAIL_PCT
    position_pct: float = APPLE_TRADER_POSITION_PCT
    flatten_before_close_min: int = APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN


def config_signature(
    config: "AppleTraderConfig | None" = None, model_threshold: "float | None" = None
) -> str:
    """Compact identity of one rule set, standing in for a model name.

    Two runs of this agent differ only in these numbers and the model behind
    them, so SimLab groups and de-duplicates runs on this string exactly as it
    does on `provider/model` for the LLM agents -- retuning the trailing stop,
    or swapping the classifier for the forecaster, is a new configuration to
    test rather than a repeat of one already tested. The model leads the string
    because a threshold read without it is meaningless: the two models' scales
    are unrelated.
    """
    c = config or AppleTraderConfig()
    threshold = c.prob_threshold if c.prob_threshold is not None else model_threshold
    shown = f"{threshold:g}" if threshold is not None else "model"
    return (
        f"{apple_models.get(c.model_key).key}_{TICKER}(p>={shown},"
        f"trail={c.trail_pct:g}%,size={c.position_pct:g}%)"
    )


class AppleTrader:
    """The state machine driving one run: the entry question and the trailing
    stop on a position already taken. One instance per launched agent."""

    def __init__(
        self, config: "AppleTraderConfig | None" = None, model_threshold: float = 0.5
    ) -> None:
        self.config = config or AppleTraderConfig()
        # The bundle's own cut-off, used when the config doesn't override it.
        self.model_threshold = model_threshold
        # The open long: entry price, the running peak the stop trails, and how
        # many bars it has been held.
        self.entry: "dict | None" = None
        # Timestamp of the last bar acted on, so a cycle that re-reads the same
        # bar (slow fetch, market data lag) can't buy the same change twice.
        self.last_bar_ts = None

    @property
    def prob_threshold(self) -> float:
        threshold = self.config.prob_threshold
        return self.model_threshold if threshold is None else threshold

    # --- one cycle --------------------------------------------------------

    def run_cycle(self, bundle: dict, state: AppState, tracker: DecisionTracker) -> str:
        """Read the newest closed bar and act on it. Returns a short outcome
        tag ("bought", "sold", "hold", "warming_up", "closed", "no_data")."""
        sym_state = state.sym(TICKER)
        if sym_state is None:
            _log(state, {"type": "error", "text": f"{TICKER} is not being streamed; nothing to trade."})
            return "no_data"

        if not market_hours.is_market_open():
            _log(
                state,
                {"type": "status", "text": "Market closed -- Apple Trader is not watching bars."},
            )
            return "closed"

        frame = persistence_model.minute_frame(sym_state)
        read = persistence_model.read_latest(bundle, frame) if len(frame) else None
        if read is None:
            # Either the frame is too short to run the pipeline at all, or the
            # newest bar has no momentum yet -- the trailing window needs the
            # session's first `horizon` minutes before it produces a number.
            _log(
                state,
                {"type": "status", "text": f"No scoreable {TICKER} bar yet; momentum is still forming."},
            )
            return "no_data"

        position = tracker.position_for(TICKER)
        fresh_bar = read["ts"] != self.last_bar_ts
        if fresh_bar:
            self.last_bar_ts = read["ts"]

        if position > 0 and self.entry is None:
            # A position without a remembered entry (agent restarted onto an
            # existing ledger): adopt it at the current price, so the trailing
            # stop starts from here rather than from a peak it never saw.
            self.entry = {"price": read["price"], "peak": read["price"], "bars": 0}
        if position <= 0:
            self.entry = None

        if fresh_bar and self.entry is not None:
            self.entry["bars"] += 1
            # The stop trails the highest price the position has TRADED at, so
            # the peak comes from the bar's high, not its close.
            self.entry["peak"] = max(self.entry["peak"], read["high"])

        _log(state, {"type": "analysis", "text": self._read_summary(read, position)})

        if position > 0:
            reason = self._exit_reason(read)
            if reason is not None:
                self._sell(state, tracker, position, read, reason)
                return "sold"
            return "hold"

        if fresh_bar and self._entry_signal(read):
            if self._closing_soon():
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"Entry signal on the {read['ts']:%H:%M} bar, but the session is "
                            f"inside its last {self.config.flatten_before_close_min} min and "
                            "any position would be flattened straight back out. Standing down."
                        ),
                    },
                )
                return "hold"
            return "bought" if self._buy(state, tracker, read) else "hold"
        return "warming_up" if read["warming_up"] else "hold"

    def _entry_signal(self, read: dict) -> bool:
        """A change into the positive regime the model expects to hold.

        `proba` is None on every bar that is not such a change, and also on one
        that is but whose 20-bar feature window is incomplete -- in which case
        the model is not asked, and an unasked model is not a yes. Exactly the
        bars `mshift.backtest._signal_sequences` builds a sequence for.
        """
        return read["to_positive"] and read["proba"] is not None and (
            read["proba"] >= self.prob_threshold
        )

    def _closing_soon(self) -> bool:
        """Whether the flatten-before-close rule is already in force."""
        to_close = market_hours.seconds_to_close()
        return to_close is not None and to_close <= self.config.flatten_before_close_min * 60

    # --- the check on an open position -------------------------------------

    def _exit_reason(self, read: dict) -> "str | None":
        """Why this long should be closed on this bar, or None to keep holding."""
        entry = self.entry or {}
        entry_price = entry.get("price") or 0.0
        peak = entry.get("peak") or entry_price
        price = read["price"]
        pnl_pct = (price / entry_price - 1) * 100 if entry_price else 0.0
        drawdown_pct = (price / peak - 1) * 100 if peak else 0.0

        if self.config.trail_pct and drawdown_pct <= -abs(self.config.trail_pct):
            return (
                f"Trailing stop: ${price:,.2f} is {drawdown_pct:+.2f}% off the ${peak:,.2f} "
                f"high since the ${entry_price:,.2f} entry, past the "
                f"{self.config.trail_pct:.2f}% give-back. Selling at market ({pnl_pct:+.2f}%)."
            )

        if self._closing_soon():
            to_close = market_hours.seconds_to_close() or 0.0
            return (
                f"Session ends in {to_close / 60:.0f} min. Momentum, regimes and every model "
                "feature are intraday, so the position is flattened rather than carried "
                f"overnight ({pnl_pct:+.2f}%)."
            )
        return None

    # --- orders ------------------------------------------------------------

    def _buy(self, state: AppState, tracker: DecisionTracker, read: dict) -> bool:
        """Deploy `position_pct` of the cash balance; False if it buys nothing."""
        price = read["price"]
        cash = tracker.snapshot()["cash"]
        budget = cash * self.config.position_pct / 100.0
        quantity = math.floor(budget / price * 1e4) / 1e4 if price > 0 else 0.0
        if quantity <= 0:
            _log(
                state,
                {"type": "status", "text": f"Entry signal confirmed but ${cash:,.2f} cash buys no {TICKER}."},
            )
            return False

        dwell = read["pre_dwell"]
        reasoning = (
            f"Momentum regime turned "
            f"{persistence_model.regime_name(read['prev_regime'])} -> positive on this bar "
            f"(momentum {read['mom']:+.2f}"
            + (f", the old regime had held {dwell} bars" if dwell is not None else "")
            + f"), and the persistence model puts it at {read['proba']:.0%} "
            f"(>= {self.prob_threshold:.0%}) to hold. Buying; the exit is a "
            f"{self.config.trail_pct:.2f}% trailing stop from here."
        )
        decision = tracker.record_trade(
            TICKER, "buy", quantity, reasoning, state.api_key, state.api_secret, state.feed
        )
        self._log_decision(state, decision, read)
        if decision.status == "filled":
            # The peak starts at the fill, not at this bar's high: the stop
            # trails the highest price seen since the position existed, and
            # this bar's high happened before it did.
            self.entry = {"price": decision.price, "peak": decision.price, "bars": 0}
        return decision.status == "filled"

    def _sell(
        self, state: AppState, tracker: DecisionTracker, quantity: float, read: dict, reasoning: str
    ) -> None:
        decision = tracker.record_trade(
            TICKER, "sell", quantity, reasoning, state.api_key, state.api_secret, state.feed
        )
        self._log_decision(state, decision, read)
        if decision.status == "filled":
            self.entry = None

    # --- logging -----------------------------------------------------------

    def _read_summary(self, read: dict, position: float) -> str:
        parts = [
            f"{TICKER} {read['ts']:%H:%M} ${read['price']:,.2f}",
            f"momentum {read['mom']:+.2f} "
            f"({persistence_model.regime_name(read['regime'])})",
        ]
        if read["regime_change"]:
            dwell = read["pre_dwell"]
            parts.append(
                f"regime change {persistence_model.regime_name(read['prev_regime'])} -> "
                f"{persistence_model.regime_name(read['regime'])}"
                + (f" after {dwell} bars" if dwell is not None else "")
            )
        if read["to_positive"]:
            proba = read["proba"]
            parts.append(
                f"persistence {proba:.0%} vs {self.prob_threshold:.0%}"
                if proba is not None
                else "persistence not scoreable (feature window not warm)"
            )
        if read["warming_up"]:
            parts.append(f"warming up ({read['bars_today']} bars) -- not trading")
        if position > 0 and self.entry:
            entry_price = self.entry["price"]
            pnl = (read["price"] / entry_price - 1) * 100 if entry_price else 0.0
            peak = self.entry["peak"]
            give_back = (read["price"] / peak - 1) * 100 if peak else 0.0
            parts.append(
                f"long {position:g} sh @ ${entry_price:,.2f} ({pnl:+.2f}%), "
                f"{self.entry['bars']} bars, peak ${peak:,.2f} ({give_back:+.2f}% off, "
                f"stop at -{self.config.trail_pct:.2f}%)"
            )
        return " · ".join(parts)

    def _log_decision(self, state: AppState, decision, read: dict) -> None:
        _log(
            state,
            {
                "type": "decision",
                "action": decision.action,
                "symbol": decision.symbol,
                "status": decision.status,
                "price": decision.price,
                "quantity": decision.filled_quantity,
                "reasoning": decision.reasoning,
                "regime": persistence_model.regime_name(read["regime"]),
            },
        )


# --- the loop ---------------------------------------------------------------

def _seconds_to_next_bar(cycle_sec: int, lag: float = APPLE_TRADER_BAR_LAG_SEC) -> float:
    """Seconds to wait so the next cycle lands just after a bar closes.

    Aligning to the clock (rather than sleeping a flat `cycle_sec` from
    wherever the last cycle finished) keeps one cycle per closed bar however
    long the scoring itself took.
    """
    ts = _now().timestamp()
    return (math.floor(ts / cycle_sec) + 1) * cycle_sec + lag - ts


def _apple_trader_loop(
    state: AppState,
    tracker: DecisionTracker,
    config: AppleTraderConfig,
    cycle_sec: int,
    stop_event: threading.Event,
) -> None:
    model = apple_models.get(config.model_key)
    bundle = apple_models.load(config.model_key)
    if bundle is None:
        _log(
            state,
            {
                "type": "error",
                "text": (
                    f"{apple_models.unavailable_reason(config.model_key)} "
                    f"Apple Trader cannot run without it."
                ),
            },
        )
        scoring.end_session(state, tracker)
        state.agent_running = False
        return

    trader = AppleTrader(config, model_threshold=persistence_model.model_threshold(bundle))
    metrics = bundle.get("metrics") or {}
    _log(
        state,
        {
            "type": "status",
            "text": (
                f"Apple Trader armed on {model.label} (fitted "
                f"{bundle.get('trained_at', 'unknown')}, held-out AUC "
                f"{metrics.get('roc_auc', float('nan')):.2f}): watching every {TICKER} minute "
                f"bar for a regime change into positive momentum, buying one the model rates "
                f"at least {trader.prob_threshold:.0%} likely to persist, and exiting on a "
                f"{config.trail_pct:.2f}% trailing stop from the high since entry."
            ),
        },
    )

    while not stop_event.is_set():
        try:
            trader.run_cycle(bundle, state, tracker)
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Apple Trader cycle failed: {exc}"})
        scoring.maybe_score_day(state, tracker)
        stop_event.wait(_seconds_to_next_bar(cycle_sec))

    scoring.end_session(state, tracker)
    state.agent_running = False
    _log(state, {"type": "status", "text": "Apple Trader stopped"})
    obs.flush()


def launch_apple_trader(
    state: AppState,
    tracker: DecisionTracker,
    config: "AppleTraderConfig | None" = None,
    cycle_sec: int = APPLE_TRADER_CYCLE_SEC,
) -> None:
    """Stop any running agent for this state, then start the Apple Trader loop.

    It trades only `TICKER`, which must already be streamed. No LLM client, no
    tools and no tactics are involved -- the loop places its own orders through
    the same `DecisionTracker` as every other personality.
    """
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    scoring.begin_session(state, APPLE_TRADER_KEY, [TICKER])
    threading.Thread(
        target=_apple_trader_loop,
        args=(state, tracker, config or AppleTraderConfig(), cycle_sec, stop_event),
        daemon=True,
    ).start()
