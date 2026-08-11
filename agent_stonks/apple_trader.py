"""Apple Trader -- a rule-based agent with no LLM in the loop.

Every other personality in `agent_stonks.agent` is a system prompt handed to a
model that reasons its way to a decision. This one is a plain loop: once a
minute it looks at the AAPL bar that just closed and asks a saved model from
FinNotebooks one question. Same paper ledger, same fill path, same log -- only
the decision-making is deterministic, so the same tape always produces the same
trades.

Two strategies live here, and `AppleTraderConfig.model_key` picks between them
by picking a model (see `agent_stonks.apple_models`, which owns that mapping):

* **the momentum rules** (`persistence`, `nbeats`, from TimeToChange2) -- buy a
  momentum-regime change into positive that the model expects to hold, sell on
  a trailing stop or a forecast reversal. One question per bar, all day. This
  is the original agent and everything below down to "Against the notebook"
  describes it.
* **the day-range rules** (`dayrange`, from TimeToChange3) -- one forecast at
  9:35 of where the session's high and low will land, then two resting levels
  derived from it and nothing more asked of the model all day. See
  `DayRangeTrader`.

They share the ledger, the sizing, the flatten-before-close rule and the config
record, and nothing else -- a forecast of the day's high is not a probability
and there is no threshold to compare it against. `build_trader` is the seam.

---------------------------------------------------------------------------
The momentum rules
---------------------------------------------------------------------------

Two settings decide what that question is. `AppleTraderConfig.entry_mode`
chooses whether the loop buys the regime change it can *see* on the bar
(`confirm`) or the one the model *predicts* for the next bar (`anticipate`,
the default) -- the difference between entering after the momentum score has
crossed its threshold and entering before. `model_key` chooses who answers: the
incumbent persistence classifier, or the N-BEATS ensemble that forecasts
momentum and derives its probability from 500 sampled futures. The two settings
are not independent -- only a forecaster can answer the anticipation question
-- but both are part of a run's configuration identity rather than a different
agent.

The rules
---------
* BUY on one bar, one question, one answer -- but *which* bar and which
  question is `AppleTraderConfig.entry_mode`:

  - `anticipate` (default): the last closed bar's regime is still **negative or
    balanced**, and the model's forecast puts it at `prob_threshold` or better
    to turn positive on the next bar and stay positive. The regime has not
    turned yet; the trade is taken on the prediction that it is about to.
  - `confirm`: the last closed bar **is** a momentum-regime change into
    positive, and the model's persistence probability for it is at least
    `prob_threshold`. There is no confirmation count on top, because the thing
    a confirmation window would wait for -- the new regime surviving -- is
    exactly what the model is being asked.

  The one thing that overrides a signal either way is the clock: no position is
  opened inside the closing flatten window, because a long the next rule is
  about to shut is not a trade, it is two commissions.
* SELL on either of two triggers, whichever comes first:

  - **the trailing stop**: price has fallen `trail_pct` below the highest price
    seen since the entry. The peak ratchets up and never down, so the rule
    starts `trail_pct` under the entry and turns into a profit lock as the move
    runs.
  - **the forecast reversal** (`reversal_threshold`, None to switch it off):
    the model puts the positive regime at that probability or better of
    flipping to **negative** somewhere inside its 15-bar forecast horizon.

  The two are deliberately different kinds of rule. The trailing stop is
  backward-looking and unconditional: it waits for the give-back to actually
  happen, and it is the only thing that can save a position the model is wrong
  about. The reversal exit is the same forecast the entry was taken on, read in
  the other direction -- it can leave while price is still at its high, which
  is the entire point, and it can also be wrong in the one way the stop cannot
  be, by selling a move that goes on without it. Neither subsumes the other,
  so both run and either one closes the book.

Why the reversal exit asks about *negative* and not "no longer positive"
------------------------------------------------------------------------
A regime that fades from positive to balanced is the ordinary shape of a move
pausing; the Schmitt trigger's whole purpose is that `mom` dropping below
`exit_threshold` is not the same event as it crossing `-enter_threshold`.
Selling on the fade would exit most winners mid-move and duplicate what the
trailing stop already does more cheaply. Selling on the flip is a claim the
model is uniquely placed to make and that price has not made yet.

The flip is checked anywhere inside the horizon rather than on the next bar,
because positive to negative in one bar requires momentum to fall from above
`exit_threshold` to below `-enter_threshold` at once -- a rule asking only
about the next bar would essentially never fire. That makes the threshold the
only thing standing between "the model sees some risk" and a sell, and unlike
`prob_threshold` it has no notebook behind it: nothing in TimeToChange2 ever
grid-searched an exit. `config.APPLE_TRADER_REVERSAL_THRESHOLD` carries the
measurement the shipped default was picked from.

What the rule actually fires on
-------------------------------
Worth being blunt about, because the name oversells it. Across five AAPL
sessions (2026-07-27 SIP, 2026-08-03..06 yfinance) there are 21 positive-regime
runs and **not one of them ends in negative** -- every single one decays to
balanced first. On this tape the literal event the rule is named after does not
happen.

What the number is doing, then, is reading the lower tail of the forecast fan:
it rises when enough sampled futures fall far enough to cross `-enter_threshold`
that the regime is visibly fragile, and empirically those are the bars near the
end of the run. Over 428 held bars it separates "within 3 bars of the run
ending" from "8+ bars still to go" at 0.89 AUC, and at the shipped 0.30 it
fires about twice a session with 55% of those firings landing near the end,
against a 14.7% base rate.

So read it as **an early warning that the positive regime is running out**, not
as a prediction that price is about to trend down. That is still the useful
thing for an exit -- the alternative framing, "will the regime stop being
positive", scores a higher 0.955 AUC and is completely unusable as a rule,
because 62% of all held bars clear 0.5 on it. Fading to balanced is what a move
does constantly; it is not news. The strict definition is narrow on purpose,
and its narrowness is what makes a threshold mean something.

And whether it makes money is, so far, unknown
----------------------------------------------
Being able to see the run ending is not the same as being paid for it, and the
only A/B run to date does not show that it is. Same model, same
`anticipate,p>=0.05,trail=0.5%` rules, the exit off and then on:

    2026-07-27 (SIP, 1 session)      off +0.140%   0.30 +0.042%   0.20 +0.293%
    2026-08-03..06 (yf, 4 sessions)  off -0.577%   0.30 -0.637%   0.20 -0.818%

Six round trips and twelve. That is noise, and it is quoted here so nobody
reads the 0.89 AUC above as a result about P&L -- the rule demonstrably fires
where it is supposed to and has not yet been shown to help. The mechanism it
loses on is visible in the trades: selling early frees the book, and a freed
book takes the *next* entry, so an exit change rewrites the rest of the
session's trades rather than just closing one of them earlier. That cuts both
ways and neither way is measured yet. `reversal_threshold=None` is a real
setting, not a legacy path, for exactly this reason.

Like `anticipate`, this rule is a question about a bar that is not a regime
change, so only a forecasting model can be asked it; pairing it with the
incumbent classifier fails at launch (`reversal_exit_error`) rather than
holding every position to the trailing stop and looking like a rule that simply
never triggered.

Why `anticipate` is the default
-------------------------------
`confirm` is the rule TimeToChange2's simulator runs, and on a live tape it is
structurally late. The regime turns positive when the smoothed momentum score
crosses `enter_threshold` (0.90), and that score is a 15-bar return smoothed
over 7 more -- so by the bar the trigger fires, the move that produced it is
fifteen to twenty bars old. Measured over the four to-positive changes on the
2026-07-27 SIP tape: price had already run 0.13-0.24% into the signal, while
the *forward* 30-bar excursion from the signal was 0.13-0.23%. Against a 0.50%
trailing stop that is a strategy buying the end of the move -- two of that
session's three round-trips lost and it finished -0.41%.

`anticipate` asks the same model the same question one bar earlier, which is
the earliest a forecast can be checked at all: the horizon is 15 bars and the
persistence label wants 15 bars of survival, so only a turn on the very next
forecast bar leaves room to verify it holds. It fires while the regime is still
balanced or negative -- which is the point -- and it is selective rather than
chatty: on that tape the question is posed on 286 bars and only 7 clear 0.05,
against a median of 0.000. The bars leading into the four real changes score
0.43, 0.09, 0.00 and 0.20 -- the third being the change whose old regime had
held 8 bars, which the dwell gate zeroes in both modes. Entering the same three
episodes one to six bars early took the session to +0.08%; one day and three
trades is a sanity check on the wiring, not evidence about the edge.

Both modes carry the same observable gate (the regime being left must already
have run `min_dwell` bars), so `prob_threshold` means the same kind of thing in
each. It is not, however, *tuned* for `anticipate`: the cut-off a bundle ships
was grid-searched on the confirmation question, so treat it as a starting point
and re-tune it in SimLab rather than as a validated setting.

`anticipate` needs a model that can forecast, so it runs on the N-BEATS
ensemble and not on the incumbent classifier, which was fitted on change bars
and has nothing to say about a bar that is not one. Pairing them fails at
launch (`entry_mode_error`) rather than producing an empty ledger.

Nothing else closes the position except the closing bell: momentum, regimes and
every feature the model uses are intraday and do not survive the overnight gap,
so the book is flattened `flatten_before_close_min` before the close. That last
rule is also why the reversal exit is not asked to look further than it does --
a warning about a flip fifteen bars out is worth acting on at 11:00 and is
nearly moot at 15:50.

Against the notebook
--------------------
The `confirm` entry is the rule TimeToChange2's own simulator runs
(`mshift.backtest.simulate_day`): the same to-positive-change trigger, the same
`proba >= threshold` test on the same 20-bar window, and the same refusal to
open a position it is about to be forced out of (there, "no decision on the
last bar of the session"). `tests/test_persistence_model.py` pins the feature
pipeline to `mshift` bar for bar, so a signal in that mode is a signal there.
Everything in this section is about that mode; `anticipate` has no counterpart
in the notebook, which only ever scores change bars.

Two differences remain, deliberately:

* the **exit**. The notebook sells when *actual* momentum drops below a level.
  Here a trailing stop does that job on price, and -- when it is armed -- the
  reversal rule does something the notebook has no analogue for: it sells on
  *forecast* momentum, before the level is reached. Different rules, different
  holding times, and because a held position blocks the next entry, that alone
  can change which later signals become trades.
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
veto".

The exit is where that reading matters most, because the reversal rule leans on
the model in a way nothing else here does. Every AUC quoted above is measured on
the *entry* question -- does a change into positive persist -- scored on regime
change bars. The reversal rule asks a question no fold ever scored: on an
ordinary positive bar, does this regime flip negative. It is the same forecast
underneath, so it inherits the ensemble's honest 15-bar skill, but "the
forecaster is better than chance at ranking to-positive changes" is not evidence
that it times exits. Whether this exit beats holding to the trailing stop is an
open question, and SimLab is where it gets answered -- run the same dataset with
`reversal_threshold` set and cleared and compare, which is exactly what
`config_signature` keeps as two configurations.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from . import agent as agent_mod
from . import apple_models, historical, market_hours, persistence_model, scoring
from . import observability as obs
from .agent import _log, stop_agent
from .clock import now as _now
from .config import (
    APPLE_TRADER_BAR_LAG_SEC,
    APPLE_TRADER_BUY_K,
    APPLE_TRADER_CYCLE_SEC,
    APPLE_TRADER_ENTRY_MODE,
    APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN,
    APPLE_TRADER_MODEL,
    APPLE_TRADER_POSITION_PCT,
    APPLE_TRADER_PROB_THRESHOLD,
    APPLE_TRADER_REVERSAL_THRESHOLD,
    APPLE_TRADER_SELL_K,
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

# The two entry triggers. See the module docstring for what separates them and
# `agent_stonks.config` for why `anticipate` is the default.
ENTRY_ANTICIPATE = "anticipate"
ENTRY_CONFIRM = "confirm"
ENTRY_MODES = (ENTRY_ANTICIPATE, ENTRY_CONFIRM)

# How the two modes are named and explained wherever they are offered. Both the
# live dashboard and SimLab present this choice, and it is the setting most
# likely to be misread as cosmetic -- so the wording lives here once rather
# than drifting between two panels.
ENTRY_MODE_LABEL = {
    ENTRY_ANTICIPATE: "Anticipate the turn",
    ENTRY_CONFIRM: "Confirm the turn",
}
ENTRY_MODE_SUMMARY = {
    ENTRY_ANTICIPATE: (
        "Buys while the regime is still negative or balanced, on the model's forecast "
        "that it turns positive on the next bar and stays there. Needs a forecasting "
        "model."
    ),
    ENTRY_CONFIRM: (
        "Buys the bar the change into positive prints on, if the model rates it likely "
        "to hold. This is the notebook's rule — and by that bar the momentum score has "
        "already crossed its threshold, so the entry lands after the move that produced "
        "the signal."
    ),
}
ENTRY_MODE_PROB_LABEL = {
    ENTRY_ANTICIPATE: "Turn probability to buy",
    ENTRY_CONFIRM: "Persistence probability to buy",
}


@dataclass
class AppleTraderConfig:
    """Tunables of the loop, for both strategies.

    `model_key` comes first because it decides which of the others are even
    read: it names the model, and a model names its rule set. On the momentum
    strategy `entry_mode`, `prob_threshold`, `trail_pct` and
    `reversal_threshold` are the four that change what the agent does; on the
    day-range strategy it is `buy_k` and `sell_k`, and the momentum four are
    inert. `position_pct` and `flatten_before_close_min` are sizing and
    housekeeping and apply to both.

    Fields that do not apply are kept rather than split into two dataclasses
    so that one config survives the round trip through SimLab's JSON
    experiment record whichever model wrote it, and so switching models in the
    UI does not lose the other strategy's settings. What keeps that from
    becoming a lie is `config_signature`, which renders only the fields in
    force -- an inert `trail_pct` never reaches Results and never splits one
    strategy's runs into two configurations.
    """

    # Which saved model the agent runs on -- a key of `apple_models.MODELS`.
    # Its `strategy` decides the rules; see `build_trader`.
    model_key: str = APPLE_TRADER_MODEL
    # Which question to ask it: buy the predicted turn, or the confirmed one.
    # `anticipate` requires a model that can forecast, which is checked before
    # the loop starts rather than discovered on the first candidate.
    entry_mode: str = APPLE_TRADER_ENTRY_MODE
    # None -> the cut-off the chosen model picked on its own validation block,
    # which is the only setting that means the same thing across models.
    prob_threshold: Optional[float] = APPLE_TRADER_PROB_THRESHOLD
    trail_pct: float = APPLE_TRADER_TRAIL_PCT
    # The second exit: sell when the forecaster puts the positive regime at
    # this probability or better of flipping to negative. None switches the
    # rule off and leaves the trailing stop as the only way out, which is what
    # every run before this setting existed did -- and the only thing a
    # non-forecasting model can do, checked before the loop starts by
    # `reversal_exit_error`.
    reversal_threshold: Optional[float] = APPLE_TRADER_REVERSAL_THRESHOLD
    # --- the day-range strategy's two levels, in average daily ranges below
    # the predicted high. See `DayRangeTrader`; `config.APPLE_TRADER_BUY_K`
    # carries what the notebook's sweep did and did not establish about them.
    buy_k: float = APPLE_TRADER_BUY_K
    sell_k: float = APPLE_TRADER_SELL_K
    position_pct: float = APPLE_TRADER_POSITION_PCT
    flatten_before_close_min: int = APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN

    def __post_init__(self) -> None:
        if self.entry_mode not in ENTRY_MODES:
            raise ValueError(
                f"unknown entry_mode {self.entry_mode!r}; expected one of {ENTRY_MODES}"
            )
        if self.reversal_threshold is not None and not 0.0 <= self.reversal_threshold <= 1.0:
            raise ValueError(
                f"reversal_threshold {self.reversal_threshold!r} is not a probability; "
                "pass None to switch the forecast exit off"
            )
        # Validated for every config rather than only the day-range ones: the
        # ordering is what makes the rule a rule (buy below, sell above), and
        # catching it here means a UI cannot hand the loop a pair that would
        # buy and sell on the same bar forever. Checked unconditionally because
        # a momentum config carrying a nonsense pair becomes a live bug the
        # moment somebody switches its model.
        if self.sell_k >= self.buy_k:
            raise ValueError(
                f"sell_k {self.sell_k!r} must sit above the buy level, i.e. strictly "
                f"below buy_k {self.buy_k!r} — both are distances *below* the predicted "
                "high, so the smaller number is the higher price"
            )

    @property
    def strategy(self) -> str:
        """Which rule set this configuration runs."""
        return apple_models.strategy(self.model_key)

    @property
    def sells_on_reversal(self) -> bool:
        """Whether the forecast exit is armed at all."""
        return self.reversal_threshold is not None


def config_signature(
    config: "AppleTraderConfig | None" = None, model_threshold: "float | None" = None
) -> str:
    """Compact identity of one rule set, standing in for a model name.

    Two runs of this agent differ only in these numbers and the model behind
    them, so SimLab groups and de-duplicates runs on this string exactly as it
    does on `provider/model` for the LLM agents -- retuning the trailing stop,
    swapping the classifier for the forecaster, or moving the entry from the
    confirmed change to the predicted one is a new configuration to test rather
    than a repeat of one already tested. The model leads the string because a
    threshold read without it is meaningless: the two models' scales are
    unrelated. The entry mode follows it because the same model answers a
    different question in each.

    The forecast exit appears only when it is armed, so a rule set that does not
    use it signs exactly as it did before the setting existed -- runs recorded
    then and runs configured now really are the same strategy, and Results
    should go on grouping them together. For the same reason only the fields
    the chosen strategy actually reads appear at all: a day-range run signed
    with an inert trailing stop would split into two configurations the first
    time somebody moved a knob that changes nothing.
    """
    c = config or AppleTraderConfig()
    model = apple_models.get(c.model_key)
    if model.strategy == apple_models.STRATEGY_DAYRANGE:
        return (
            f"{model.key}_{TICKER}(buy=H-{c.buy_k:g}A,sell=H-{c.sell_k:g}A,"
            f"size={c.position_pct:g}%)"
        )
    threshold = c.prob_threshold if c.prob_threshold is not None else model_threshold
    shown = f"{threshold:g}" if threshold is not None else "model"
    reversal = f",rev>={c.reversal_threshold:g}" if c.sells_on_reversal else ""
    return (
        f"{model.key}_{TICKER}({c.entry_mode},p>={shown},"
        f"trail={c.trail_pct:g}%{reversal},size={c.position_pct:g}%)"
    )


def entry_mode_error(config: AppleTraderConfig, bundle: "dict | None") -> "str | None":
    """Why this rule set cannot run on this bundle, or None if it can.

    The one pairing that does not work is `anticipate` on a model that cannot
    forecast. Checked once before the loop starts rather than per bar, because
    the failure mode it prevents is the quiet one: `read_latest` leaves
    `turn_proba` as None on a classifier, `_entry_signal` reads None as "not a
    buy", and the run finishes clean with an empty ledger that looks like a
    strategy result instead of a misconfiguration.
    """
    if config.strategy != apple_models.STRATEGY_MOMENTUM:
        return None
    if config.entry_mode != ENTRY_ANTICIPATE:
        return None
    if persistence_model.anticipates(bundle):
        return None
    model = apple_models.get(config.model_key)
    return (
        f"{model.label} cannot forecast a regime change that has not happened yet, "
        f"so it cannot run the '{ENTRY_ANTICIPATE}' entry. Either switch the model "
        f"({_forecasters()}) or switch the entry to '{ENTRY_CONFIRM}'."
    )


def reversal_exit_error(config: AppleTraderConfig, bundle: "dict | None") -> "str | None":
    """Why the forecast exit cannot run on this bundle, or None if it can.

    The same shape of mistake `entry_mode_error` catches, on the way out
    instead of the way in: a positive bar that is not a regime change is not a
    question the incumbent classifier was fitted to answer. Left unchecked it
    would fail even more quietly than the entry version -- `read_latest` leaves
    `reversal_proba` as None, the exit reads None as "keep holding", and the
    run finishes with a full ledger of trades that all exited on the trailing
    stop, which is indistinguishable from the rule simply never triggering.
    """
    if config.strategy != apple_models.STRATEGY_MOMENTUM:
        return None
    if not config.sells_on_reversal:
        return None
    if persistence_model.forecasts_reversal(bundle):
        return None
    model = apple_models.get(config.model_key)
    return (
        f"{model.label} cannot forecast the breakdown of a regime, so it cannot run "
        f"the reversal exit. Either switch the model ({_forecasters()}) or clear the "
        f"reversal threshold and exit on the trailing stop alone."
    )


def config_error(config: AppleTraderConfig, bundle: "dict | None") -> "str | None":
    """The first reason this rule set cannot run on this bundle, or None.

    One call for every caller that is about to start a run, so a rule added
    later is checked everywhere it needs to be rather than in whichever launch
    path was remembered.
    """
    return entry_mode_error(config, bundle) or reversal_exit_error(config, bundle)


def _forecasters() -> str:
    """The models that can answer a question about a bar that is not a change."""
    return ", ".join(
        m.label
        for m in apple_models.MODELS.values()
        if m.strategy == apple_models.STRATEGY_MOMENTUM and m.anticipates
    )


def _dayrange():
    """`agent_stonks.dayrange_model`, imported on first use.

    Kept out of this module's imports because it pulls PyTorch and LightGBM in
    (in that order, deliberately -- see its docstring), and the two momentum
    models must not make every process that starts Apple Trader pay for a 200 MB
    dependency they do not use. `sys.modules` makes the repeat calls free.
    """
    from . import dayrange_model

    return dayrange_model


def _order_quantity(cash: float, price: float, position_pct: float) -> float:
    """Shares that `position_pct` of `cash` buys at `price`, rounded down.

    Shared by both strategies: how much to deploy is a property of the agent,
    not of the rule that decided to deploy it.
    """
    if price <= 0:
        return 0.0
    budget = cash * position_pct / 100.0
    return math.floor(budget / price * 1e4) / 1e4


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

    @property
    def reversal_threshold(self) -> float:
        """The forecast exit's cut-off. Only read behind `sells_on_reversal`,
        which is what makes the fallback unreachable rather than a default
        anybody trades on -- there is no model-chosen cut-off to fall back to
        here, because no notebook ever fitted one for an exit."""
        return self.config.reversal_threshold or 1.0

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

        position = tracker.position_for(TICKER)
        frame = persistence_model.minute_frame(sym_state)
        # The reversal question costs a full forecast on every bar it is asked
        # on, and only an open position can act on the answer -- so it is asked
        # only when both are true.
        read = (
            persistence_model.read_latest(
                bundle, frame, holding=position > 0 and self.config.sells_on_reversal
            )
            if len(frame)
            else None
        )
        if read is None:
            # Either the frame is too short to run the pipeline at all, or the
            # newest bar has no momentum yet -- the trailing window needs the
            # session's first `horizon` minutes before it produces a number.
            _log(
                state,
                {"type": "status", "text": f"No scoreable {TICKER} bar yet; momentum is still forming."},
            )
            return "no_data"

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

    def _entry_probability(self, read: dict) -> "float | None":
        """The number this entry mode is asking about, or None if this bar does
        not pose its question.

        `read_latest` fills exactly one of the two in, and only on a bar where
        the question applies and the 20-bar feature window behind it is
        complete. So neither branch has to re-check which bar it is looking at:
        an absent number is a bar with nothing to decide, and an unasked model
        is not a yes.
        """
        if self.config.entry_mode == ENTRY_ANTICIPATE:
            # Non-None only where the regime is still negative or balanced --
            # which is the entire point of this mode.
            return read["turn_proba"]
        # Non-None only on a bar that IS a change into positive, exactly the
        # bars `mshift.backtest._signal_sequences` builds a sequence for.
        return read["proba"] if read["to_positive"] else None

    def _entry_signal(self, read: dict) -> bool:
        """Whether this bar is a buy under the configured entry mode."""
        proba = self._entry_probability(read)
        return proba is not None and proba >= self.prob_threshold

    def _reversal_signal(self, read: dict) -> bool:
        """Whether the model is calling the end of the regime on this bar.

        `reversal_proba` is filled in only where the question was both armed
        and applicable -- a positive regime, an open position, a forecasting
        model -- so an absent number is a bar with nothing to say rather than a
        quiet no, exactly as on the entry side.
        """
        if not self.config.sells_on_reversal:
            return False
        proba = read["reversal_proba"]
        return proba is not None and proba >= self.reversal_threshold

    def _closing_soon(self) -> bool:
        """Whether the flatten-before-close rule is already in force."""
        to_close = market_hours.seconds_to_close()
        return to_close is not None and to_close <= self.config.flatten_before_close_min * 60

    # --- the check on an open position -------------------------------------

    def _exit_reason(self, read: dict) -> "str | None":
        """Why this long should be closed on this bar, or None to keep holding.

        Three ways out, checked in the order of how little discretion they
        leave. The trailing stop is a fact about price and comes first. The
        forecast reversal is a prediction and comes second, so a bar where both
        fire is reported as the stop it actually was. The closing bell is last
        because it is not about this position at all.
        """
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

        if self._reversal_signal(read):
            return (
                f"Forecast reversal: the momentum regime is still "
                f"{persistence_model.regime_name(read['regime'])} (momentum "
                f"{read['mom']:+.2f}, {read['bars_in_regime']} bars in), but the model puts "
                f"it at {read['reversal_proba']:.0%} (>= {self.reversal_threshold:.0%}) to "
                f"flip negative inside the forecast horizon. Selling into the regime the "
                f"trade was taken on rather than waiting for the give-back "
                f"({pnl_pct:+.2f}%)."
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
        quantity = _order_quantity(cash, price, self.config.position_pct)
        if quantity <= 0:
            _log(
                state,
                {"type": "status", "text": f"Entry signal confirmed but ${cash:,.2f} cash buys no {TICKER}."},
            )
            return False

        reasoning = self._entry_reasoning(read)
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

    def _entry_reasoning(self, read: dict) -> str:
        """Why this bar was bought, in the terms of the mode that bought it.

        The two modes buy on opposite sides of the same event, so a single
        sentence covering both would have to be vague about the one thing a
        reader of the ledger most needs to know: whether the regime had already
        turned when the order went in.
        """
        trail = f"the exit is a {self.config.trail_pct:.2f}% trailing stop from here."
        if self.config.entry_mode == ENTRY_ANTICIPATE:
            dwell = read["bars_in_regime"]
            return (
                f"Momentum regime is still "
                f"{persistence_model.regime_name(read['regime'])} on this bar "
                f"(momentum {read['mom']:+.2f}, and it has held {dwell} bars), but the "
                f"forecast puts it at {read['turn_proba']:.0%} "
                f"(>= {self.prob_threshold:.0%}) to turn positive on the next bar and "
                f"stay there. Buying the turn before it prints; {trail}"
            )
        dwell = read["pre_dwell"]
        return (
            f"Momentum regime turned "
            f"{persistence_model.regime_name(read['prev_regime'])} -> positive on this bar "
            f"(momentum {read['mom']:+.2f}"
            + (f", the old regime had held {dwell} bars" if dwell is not None else "")
            + f"), and the persistence model puts it at {read['proba']:.0%} "
            f"(>= {self.prob_threshold:.0%}) to hold. Buying; {trail}"
        )

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
        if self.config.entry_mode == ENTRY_ANTICIPATE:
            turn = read["turn_proba"]
            if turn is not None:
                parts.append(f"turns positive {turn:.0%} vs {self.prob_threshold:.0%}")
        elif read["to_positive"]:
            proba = read["proba"]
            parts.append(
                f"persistence {proba:.0%} vs {self.prob_threshold:.0%}"
                if proba is not None
                else "persistence not scoreable (feature window not warm)"
            )
        if read["warming_up"]:
            parts.append(f"warming up ({read['bars_today']} bars) -- not trading")
        reversal = read["reversal_proba"]
        if reversal is not None:
            parts.append(f"flips negative {reversal:.0%} vs {self.reversal_threshold:.0%}")
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


class DayRangeTrader:
    """The day-range rules: one forecast at 9:35, then two resting levels.

    TimeToChange3's model says where the session's high and low will land, and
    it says it exactly once -- from the daily history up to yesterday plus the
    first five minutes of this morning. Nothing about the forecast updates
    intraday, so the rule built on it cannot be a per-bar signal. It is two
    price levels, set at 9:35 and held all day, from notebook 05:

        buy_level  = H - buy_k  * A
        sell_level = H - sell_k * A

    with `H` the predicted high and `A` the 14-day average daily range in
    dollars. Buy when the price comes down to the buy level, sell when it comes
    back up to the sell level, repeat as often as the day allows, and flatten
    what is still open before the close.

    It is a mean-reversion bet, and the reason it is shaped that way is in the
    model's own results: what TimeToChange3 forecasts well is the *width* of
    the session, not its direction. So the rule never takes a view on where the
    day is going -- it buys well below where the day is expected to top out and
    sells just under it, and on a day that never dips that far it simply does
    nothing.

    What this is not
    ----------------
    Not a momentum agent with a different model bolted on. There is no regime,
    no probability, no threshold and no trailing stop; `prob_threshold`,
    `entry_mode`, `trail_pct` and `reversal_threshold` are inert here and
    `config_signature` leaves them out for that reason. The two things that
    change what it does are `buy_k` and `sell_k`.

    Against the notebook
    --------------------
    The trigger is the notebook's, bar for bar: a buy fires on a bar whose
    **low** touched the buy level and a sell on a bar whose **high** touched
    the sell level, checked in that order and never both on one bar.
    `tests/test_dayrange_model.py` pins the forecast itself to notebook 05's
    recorded number for 2026-08-07.

    Three differences remain, and only the first is small:

    * **the fill**. The notebook rests limit orders and fills a buy at
      `min(bar_open, buy_level)` -- a touch fills *at* the level. Live there is
      no resting order in this ledger: the loop sees the bar after it closed
      and sends a market order, which fills near that bar's close. On a bar
      that dipped to the level and recovered, the notebook buys at the level
      and this buys higher. That is a real cost and it runs one way, against
      the strategy; it is the price of the paper ledger being market-order
      only, not a modelling choice. Read a SimLab result against the notebook's
      with that in mind.
    * **the flatten**. The notebook closes at the 15:59 bar, so the whole
      session is available. Here `flatten_before_close_min` applies as it does
      to every other agent -- at the default of 5 the position is closed around
      15:55, giving up the last few minutes. Set it to 1 for the notebook's
      behaviour.
    * **no entry inside the flatten window**. The notebook has no such rule
      because it never needs one; here a long the next rule is about to shut is
      two commissions, exactly as on the momentum side.

    And what the rule is worth, honestly: notebook 05 ran it across all 21
    sessions that had minute data, 15 of which traded, 10 profitable, $949
    total on $10,000 a day. Twenty-one sessions is a sanity check, not an edge,
    and there are no commissions or slippage in that number.
    """

    def __init__(self, config: "AppleTraderConfig | None" = None) -> None:
        self.config = config or AppleTraderConfig()
        # The session's forecast and the two levels derived from it, or None
        # before 9:35. Keyed by date so a multi-day run re-forecasts each
        # morning rather than trading Tuesday off Monday's levels.
        self.plan: "dict | None" = None
        # Why today has no plan, when the reason is permanent for the session
        # (too little daily history, an opening window that cannot be
        # recovered). Kept so the failure is logged once instead of every
        # minute for six and a half hours.
        self.blocked: "dict | None" = None
        self.entry: "dict | None" = None
        self.last_bar_ts = None

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

        today = _dayrange().market_date()
        self._roll_session(today)

        frame = persistence_model.minute_frame(sym_state)
        if not len(frame):
            _log(state, {"type": "status", "text": f"No {TICKER} bars yet today."})
            return "no_data"

        want = _dayrange().opening_minutes(bundle)
        if self.plan is None:
            if self.blocked is not None:
                return "no_data"
            if len(frame) < want:
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"{len(frame)} of the first {want} {TICKER} minutes are in; the "
                            "day's high cannot be forecast until the opening window closes."
                        ),
                    },
                )
                return "warming_up"
            if not self._plan_session(bundle, state, frame, today, want):
                return "no_data"

        last = frame.iloc[-1]
        ts = frame.index[-1]
        fresh_bar = ts != self.last_bar_ts
        if fresh_bar:
            self.last_bar_ts = ts

        position = tracker.position_for(TICKER)
        if position > 0 and self.entry is None:
            # A position without a remembered entry (agent restarted onto an
            # existing ledger): adopt it, so the log reads honestly. The exit
            # is a fixed price level rather than a trailing one, so unlike the
            # momentum rules nothing about the decision depends on this.
            self.entry = {"price": float(last["close"]), "bars": 0}
        if position <= 0:
            self.entry = None
        if fresh_bar and self.entry is not None:
            self.entry["bars"] += 1

        _log(state, {"type": "analysis", "text": self._read_summary(last, ts, position)})

        # Trading starts after the opening window, since the forecast does not
        # exist before it -- the notebook's `start_after`. The bar the plan was
        # built on is the last bar *of* that window, so it is never traded.
        if ts <= self.plan["opening_end"]:
            return "warming_up"

        if position > 0:
            reason = self._exit_reason(last)
            if reason is not None:
                self._sell(state, tracker, position, last, reason)
                return "sold"
            return "hold"

        if fresh_bar and float(last["low"]) <= self.plan["buy_level"]:
            if self._closing_soon():
                _log(
                    state,
                    {
                        "type": "status",
                        "text": (
                            f"The {ts:%H:%M} bar traded down to the buy level, but the session "
                            f"is inside its last {self.config.flatten_before_close_min} min and "
                            "any position would be flattened straight back out. Standing down."
                        ),
                    },
                )
                return "hold"
            return "bought" if self._buy(state, tracker, last) else "hold"
        return "hold"

    def _roll_session(self, today) -> None:
        """Forget yesterday's forecast at the start of a new session."""
        stale_plan = self.plan is not None and self.plan["date"] != today
        stale_block = self.blocked is not None and self.blocked["date"] != today
        if stale_plan:
            self.plan = None
            self.entry = None
            self.last_bar_ts = None
        if stale_block:
            self.blocked = None

    # --- the forecast ------------------------------------------------------

    def _plan_session(
        self, bundle: dict, state: AppState, frame, today, want: int
    ) -> bool:
        """Forecast the day and set the two levels. False if it cannot be done.

        Every failure here is fatal for the session rather than for the bar --
        a daily history that is too short at 9:35 is still too short at 14:00
        -- so it is recorded in `self.blocked` and reported once.
        """
        try:
            opening = self._opening_window(state, frame, want)
            dayrange = _dayrange()
            history = dayrange.daily_frame_from_bars(
                historical.fetch_daily_ohlc_bars(
                    TICKER, days=dayrange.DAILY_HISTORY_DAYS
                )
            )
            forecast = dayrange.forecast_session(
                bundle, history, opening, today,
                open_price=historical.fetch_session_open(TICKER),
            )
        except Exception as exc:
            self.blocked = {"date": today, "reason": str(exc)}
            _log(
                state,
                {
                    "type": "error",
                    "text": (
                        f"Apple Trader cannot forecast today's {TICKER} range, so it will not "
                        f"trade this session: {exc}"
                    ),
                },
            )
            return False

        adr = forecast["adr14_abs"]
        self.plan = {
            "date": today,
            "opening_end": opening.index[-1],
            "buy_level": forecast["pred_high"] - self.config.buy_k * adr,
            "sell_level": forecast["pred_high"] - self.config.sell_k * adr,
            **forecast,
        }

        warning = _dayrange().volume_scale_warning(getattr(state, "feed", None))
        if warning:
            _log(state, {"type": "status", "text": f"Forecast caveat: {warning}"})
        _log(state, {"type": "analysis", "text": self._plan_summary()})
        return True

    def _opening_window(self, state: AppState, frame, want: int):
        """The first `want` regular-session bars of today, or a clear failure.

        The live buffer normally holds them -- it keeps the whole session -- but
        an agent started after 9:35 has a buffer that begins wherever the stream
        did, and `frame.iloc[:want]` would then hand the model five bars from
        the middle of the day as though they were the open. Rather than produce
        a confident forecast off the wrong five minutes, the 09:30 window is
        re-fetched, exactly as `agent._opening_range_for` does for the same
        reason and through the same (simulation-patched) call.
        """
        first = frame.iloc[:want]
        if float(first["minutes_from_open"].iloc[0]) < 1.0:
            return first

        open_utc = market_hours.session_open()
        window = []
        if open_utc is not None and state.api_key and state.api_secret:
            try:
                window = agent_mod.fetch_bars_window(
                    TICKER, "1Min", open_utc,
                    open_utc + timedelta(minutes=want),
                    state.api_key, state.api_secret, state.feed,
                )
            except Exception:
                window = []
        recovered = persistence_model.frame_from_bars(window)
        if len(recovered) < want or float(recovered["minutes_from_open"].iloc[0]) >= 1.0:
            raise ValueError(
                f"the first {want} minutes of the session are not in the bar buffer "
                f"(it starts at {frame.index[0]:%H:%M}) and could not be re-fetched; "
                "the forecast is built on the 09:30 window and cannot be made without it."
            )
        return recovered.iloc[:want]

    # --- the check on an open position -------------------------------------

    def _exit_reason(self, bar) -> "str | None":
        """Why this long should be closed on this bar, or None to keep holding.

        Two ways out, and neither is a stop: the level the position was opened
        to reach, and the closing bell. A day that never comes back up to the
        sell level is held to the flatten, which is the rule as specified --
        the forecast says where the day tops out, so an early exit on weakness
        would be a second, unmeasured strategy sitting on top of this one.
        """
        entry_price = (self.entry or {}).get("price") or 0.0
        price = float(bar["close"])
        pnl_pct = (price / entry_price - 1) * 100 if entry_price else 0.0

        if float(bar["high"]) >= self.plan["sell_level"]:
            return (
                f"Target: the bar traded up to ${float(bar['high']):,.2f}, at or through the "
                f"${self.plan['sell_level']:,.2f} sell level "
                f"(H − {self.config.sell_k:g} × ADR). Selling at market ({pnl_pct:+.2f}%)."
            )

        if self._closing_soon():
            to_close = market_hours.seconds_to_close() or 0.0
            return (
                f"Session ends in {to_close / 60:.0f} min and the day never came back up to "
                f"${self.plan['sell_level']:,.2f}. The forecast is a statement about today "
                f"only, so the position is flattened rather than carried overnight "
                f"({pnl_pct:+.2f}%)."
            )
        return None

    def _closing_soon(self) -> bool:
        """Whether the flatten-before-close rule is already in force."""
        to_close = market_hours.seconds_to_close()
        return to_close is not None and to_close <= self.config.flatten_before_close_min * 60

    # --- orders ------------------------------------------------------------

    def _buy(self, state: AppState, tracker: DecisionTracker, bar) -> bool:
        """Deploy `position_pct` of the cash balance; False if it buys nothing."""
        price = float(bar["close"])
        cash = tracker.snapshot()["cash"]
        quantity = _order_quantity(cash, price, self.config.position_pct)
        if quantity <= 0:
            _log(
                state,
                {"type": "status", "text": f"Buy level touched but ${cash:,.2f} cash buys no {TICKER}."},
            )
            return False

        decision = tracker.record_trade(
            TICKER, "buy", quantity, self._entry_reasoning(bar),
            state.api_key, state.api_secret, state.feed,
        )
        self._log_decision(state, decision)
        if decision.status == "filled":
            self.entry = {"price": decision.price, "bars": 0}
        return decision.status == "filled"

    def _entry_reasoning(self, bar) -> str:
        plan = self.plan
        return (
            f"The bar traded down to ${float(bar['low']):,.2f}, at or through the "
            f"${plan['buy_level']:,.2f} buy level — {self.config.buy_k:g} average daily "
            f"ranges (${plan['adr14_abs']:,.2f} each) below the ${plan['pred_high']:,.2f} "
            f"high the model forecast for today at the open. Buying the dip below where "
            f"the day is expected to top out; the exit is a resting sell at "
            f"${plan['sell_level']:,.2f}, or the closing bell."
        )

    def _sell(
        self, state: AppState, tracker: DecisionTracker, quantity: float, bar, reasoning: str
    ) -> None:
        decision = tracker.record_trade(
            TICKER, "sell", quantity, reasoning, state.api_key, state.api_secret, state.feed
        )
        self._log_decision(state, decision)
        if decision.status == "filled":
            self.entry = None

    # --- logging -----------------------------------------------------------

    def _plan_summary(self) -> str:
        plan = self.plan
        return (
            f"{TICKER} forecast for the session, from the first "
            f"{plan['opening_end']:%H:%M} minutes: high ${plan['pred_high']:,.2f}, low "
            f"${plan['pred_low']:,.2f} (yesterday's average ${plan['prev_avg']:,.2f}, "
            f"14-day average range ${plan['adr14_abs']:,.2f}). Buy at "
            f"${plan['buy_level']:,.2f} (H − {self.config.buy_k:g} × ADR), sell at "
            f"${plan['sell_level']:,.2f} (H − {self.config.sell_k:g} × ADR)."
        )

    def _read_summary(self, bar, ts, position: float) -> str:
        price = float(bar["close"])
        plan = self.plan
        parts = [
            f"{TICKER} {ts:%H:%M} ${price:,.2f}",
            f"buy ${plan['buy_level']:,.2f} ({price - plan['buy_level']:+.2f})",
            f"sell ${plan['sell_level']:,.2f} ({price - plan['sell_level']:+.2f})",
        ]
        if position > 0 and self.entry:
            entry_price = self.entry["price"]
            pnl = (price / entry_price - 1) * 100 if entry_price else 0.0
            parts.append(
                f"long {position:g} sh @ ${entry_price:,.2f} ({pnl:+.2f}%), "
                f"{self.entry['bars']} bars"
            )
        return " · ".join(parts)

    def _log_decision(self, state: AppState, decision) -> None:
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
            },
        )


def build_trader(config: AppleTraderConfig, bundle: dict):
    """The state machine this configuration's model calls for.

    The one place the strategy split turns into an object. Every launch path --
    the live loop below, SimLab's `rule_agents._build_apple` -- goes through
    here, so a third strategy is added in one place rather than in whichever
    entry points were remembered.

    Both returned objects expose `run_cycle(bundle, state, tracker)` and
    nothing else that a caller needs.
    """
    if apple_models.get(config.model_key).strategy == apple_models.STRATEGY_DAYRANGE:
        return DayRangeTrader(config)
    return AppleTrader(config, model_threshold=persistence_model.model_threshold(bundle))


# --- the loop ---------------------------------------------------------------

def _armed_summary(config: AppleTraderConfig, model, bundle: dict, trader) -> str:
    """The one line the log opens a run with: which model, and what it will do.

    Per strategy, because the two have nothing in common to summarise -- one is
    a probability against a threshold on every bar, the other is two price
    levels set once.
    """
    if config.strategy == apple_models.STRATEGY_DAYRANGE:
        metadata = bundle.get("metadata") or {}
        mae = (metadata.get("test_metrics_ensemble") or {}).get("mae_usd_mean")
        quality = f", held-out mean error ${mae:.2f}" if mae else ""
        return (
            f"Apple Trader armed on {model.label} (fitted "
            f"{bundle.get('trained_at', 'unknown')}{quality}): at 9:35 it forecasts where "
            f"today's {TICKER} high and low will land, then rests a buy "
            f"{config.buy_k:g} average daily ranges below the predicted high and a sell "
            f"{config.sell_k:g} below it, until the closing flatten."
        )
    metrics = bundle.get("metrics") or {}
    exit_rule = f"a {config.trail_pct:.2f}% trailing stop from the high since entry"
    if config.sells_on_reversal:
        exit_rule += (
            f", or on the model putting the regime at {trader.reversal_threshold:.0%} "
            "or better to flip negative"
        )
    return (
        f"Apple Trader armed on {model.label} (fitted "
        f"{bundle.get('trained_at', 'unknown')}, held-out AUC "
        f"{metrics.get('roc_auc', float('nan')):.2f}): watching every {TICKER} minute "
        f"bar for a regime change into positive momentum, buying one the model rates "
        f"at least {trader.prob_threshold:.0%} likely to persist, and exiting on "
        f"{exit_rule}."
    )


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

    mismatch = config_error(config, bundle)
    if mismatch is not None:
        _log(state, {"type": "error", "text": mismatch})
        scoring.end_session(state, tracker)
        state.agent_running = False
        return

    trader = build_trader(config, bundle)
    _log(state, {"type": "status", "text": _armed_summary(config, model, bundle, trader)})

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
