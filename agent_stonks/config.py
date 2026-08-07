from datetime import datetime, timezone

DATA_REST = "https://data.alpaca.markets"
BARS_STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
NEWS_STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
# Full regular session is 390 one-minute bars; 420 keeps the 09:30 ET open in
# the buffer through the close (plus a little premarket) so session-anchored
# reads (opening range, VWAP) never silently lose their anchor mid-afternoon.
MAX_BARS = 420
POLL_SEC = 3
CHART_POLL_SEC = 30

# REST-polling fallback for bars/trades and news, used only while the
# corresponding WebSocket stream is not connected (e.g. Alpaca's
# "connection limit exceeded" rejecting a second concurrent stream on the
# same API key/feed). REST calls aren't subject to that per-key streaming
# connection cap, so they keep working even while the socket is stuck.
FALLBACK_POLL_SEC = 15
NEWS_FALLBACK_POLL_SEC = 60

# Periodic REST backfill that repairs holes in the live bar series while the
# WebSocket IS connected: the stream never re-delivers bars that closed during
# a reconnect, and thin symbols get no bar at all for minutes without a trade
# on the subscribed feed.
BACKFILL_POLL_SEC = 60
OPTIONS_POLL_SEC = 60
OPTIONS_WALL_HISTORY_MAXLEN = 200
TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"]
FEEDS = ["iex", "sip"]

# High-volume alert: trigger when today's cumulative volume exceeds
# VOLUME_ALERT_DEFAULT_MULTIPLIER x the average daily volume. The baseline is
# the mean of the last VOLUME_ADV_WINDOW completed daily volumes; with fewer
# than VOLUME_ADV_MIN_DAYS completed days (thin history / early session), it
# falls back to yesterday's single-day volume.
VOLUME_ALERT_DEFAULT_MULTIPLIER = 1.5
VOLUME_ADV_WINDOW = 20
VOLUME_ADV_MIN_DAYS = 5

# Quote reliability thresholds for get_quote. The IEX feed reports IEX's own
# top-of-book, not the consolidated NBBO: outside regular hours or when IEX's
# book is empty near the touch, the "latest quote" is a placeholder-wide
# two-sided quote (e.g. ±5% around the mid, 100x100) or an hours-old snapshot.
# Quotes wider than QUOTE_WIDE_SPREAD_PCT percent of the mid, or older than
# QUOTE_STALE_SEC, get a warning attached so the agent doesn't treat them as
# executable prices.
QUOTE_WIDE_SPREAD_PCT = 1.0
QUOTE_STALE_SEC = 120.0

# Trading agent
AGENT_CYCLE_SEC = 60
AGENT_LOG_POLL_SEC = 4
AGENT_PERFORMANCE_POLL_SEC = 60
AGENT_EQUITY_HISTORY_MAXLEN = 5000
AGENT_MAX_TOOL_ITERS = 8
PAPER_STARTING_CASH = 100_000.0
TRADE_FIXED_COST = 1.15

# Daily agent-accuracy scoring (see agent_stonks.scoring): a scoring session
# runs at most once per UTC day, and only after the day has accumulated at
# least this much total agent runtime -- short experiments alone never score.
SCORING_MIN_TOTAL_RUNTIME_SEC = 3600

# Premarket analyst: it may start its single opening-tactics cycle no earlier
# than PREMARKET_LEAD_SEC before the opening bell; while holding for that
# window it re-checks the clock every PREMARKET_WAIT_POLL_SEC.
PREMARKET_LEAD_SEC = 120
PREMARKET_WAIT_POLL_SEC = 30.0

# Apple Trader: the rule-based (non-LLM) loop that watches every closed AAPL
# minute bar for a momentum-regime change into positive and asks a saved model
# about it (see agent_stonks.apple_trader).
#
# ENTRY_MODE decides *when* it asks, and it is the setting that changes what the
# agent does most:
#   "anticipate"  buy while the regime is still negative or balanced, on the
#                 model's forecast that it turns positive on the next bar. Needs
#                 a forecasting model, so only "nbeats" can run it.
#   "confirm"     buy the bar the change has already happened on, if the model
#                 rates it likely to hold. Either model can run this, and it is
#                 what the agent did before anticipation existed -- but by then
#                 momentum has already crossed the enter threshold, so the entry
#                 lands after the move that produced the signal.
#
# MODEL names which of the saved models answers that question -- a key of
# agent_stonks.apple_models.MODELS ("nbeats", the forecast-derived one, or
# "persistence", the incumbent classifier). It defaults to the forecaster
# because the default entry mode is one only the forecaster can answer.
# PROB_THRESHOLD is the probability a candidate has to clear to be bought; None
# uses the cut-off the chosen model picked on its own validation block, which is
# the intended setting because those cut-offs are not on a shared scale (0.07
# for the classifier's posterior, 0.05 for N-BEATS' gated survival
# probability). TRAIL_PCT is the whole exit rule: sell once price is that far
# below the highest price seen since the entry.
APPLE_TRADER_ENTRY_MODE = "anticipate"
APPLE_TRADER_MODEL = "nbeats"
APPLE_TRADER_CYCLE_SEC = 60
APPLE_TRADER_PROB_THRESHOLD: "float | None" = None
APPLE_TRADER_TRAIL_PCT = 0.5
APPLE_TRADER_POSITION_PCT = 95.0
# Flatten this many minutes before the close: momentum, regimes and the model's
# whole feature set are intraday, and none of it survives the overnight gap.
APPLE_TRADER_FLATTEN_BEFORE_CLOSE_MIN = 5
# Seconds after a minute boundary to score, giving the stream time to deliver
# the bar that just closed.
APPLE_TRADER_BAR_LAG_SEC = 5.0

# TraderByChatGPT: the other rule-based (non-LLM) AAPL loop, a long-only
# intraday breakout state machine with no model behind it at all -- its entry
# is a pure confluence predicate on the just-closed bar and its exit is an ATR
# stop that ratchets up behind the trade's peak (see
# agent_stonks.openai_rule_trader).
#
# POSITION_PCT caps the notional of one position, since ATR-based risk sizing
# can ask for an unbounded quantity when volatility collapses. TRAIL_PCT is an
# optional percentage stop measured from the running peak: it can only tighten
# the ATR trail, and 0 leaves the ATR trail as the sole exit.
TRADER_BY_CHATGPT_CYCLE_SEC = 60
TRADER_BY_CHATGPT_TRAIL_PCT = 0.0
TRADER_BY_CHATGPT_POSITION_PCT = 95.0
# Flatten this many minutes before the close: every feature the entry rule
# reads is intraday, and none of it survives the overnight gap.
TRADER_BY_CHATGPT_FLATTEN_BEFORE_CLOSE_MIN = 5
# Seconds after a minute boundary to score, giving the stream time to deliver
# the bar that just closed.
TRADER_BY_CHATGPT_BAR_LAG_SEC = 5.0

# TraderByClaude: the third rule-based (non-LLM) AAPL loop, and the deliberate
# counter-thesis to TraderByChatGPT -- it buys the pullback inside an
# established uptrend rather than the breakout out of one, so its stop sits
# under the pullback low instead of a fixed ATR distance (see
# agent_stonks.claude_rule_trader).
#
# RISK_PCT is what actually sizes a position: the dollars between the fill and
# that structural stop, as a share of equity. POSITION_PCT only caps the
# notional, for when the stop is a few cents away and the risk formula asks for
# a position several times the account.
TRADER_BY_CLAUDE_CYCLE_SEC = 60
TRADER_BY_CLAUDE_RISK_PCT = 0.25
TRADER_BY_CLAUDE_POSITION_PCT = 95.0
# Flatten this many minutes before the close: the trend structure the entry
# reads is intraday, and it does not survive the overnight gap.
TRADER_BY_CLAUDE_FLATTEN_BEFORE_CLOSE_MIN = 5
# Seconds after a minute boundary to score, giving the stream time to deliver
# the bar that just closed.
TRADER_BY_CLAUDE_BAR_LAG_SEC = 5.0

# Tactics executor: the stream nudges it on every tick, so this poll is only a
# fallback cadence covering the non-stream condition fields (vix, momentum) and
# REST-fallback sessions. The momentum window is the lookback (in minutes) for
# the `momentum_pct` tactic condition field.
TACTICS_POLL_SEC = 2.0
TACTICS_MOMENTUM_WINDOW_MIN = 10

# 13:20 UTC = 09:20 ET, just before market open (09:30 ET)
SESSION_START = datetime.now(tz=timezone.utc).replace(
    hour=13, minute=20, second=0, microsecond=0
)

PALETTE: dict[str, str] = {
    "bg": "#0f1117",
    "panel": "#1a1d27",
    "grid": "#2a2d3a",
    "up": "#26c6a2",
    "down": "#ef5350",
    "text": "#e0e0e0",
    "muted": "#888",
    "accent": "#60a5fa",
    "orange": "#fb923c",
}

# Dot / marker colors per LLM-estimated news impact label, shared by the
# News tab badges and the Live chart news markers.
NEWS_IMPACT_COLORS: dict[str, str] = {
    "positive": "#26c6a2",
    "negative": "#ef5350",
    "neutral":  "#888",
    "small":    "#fb923c",
    "unknown":  "#555",
}

# News dots on the Live chart sit above the high of the minute bar containing
# the article's timestamp, offset by this fraction of the session's price range
# so the spacing looks right at any price scale.
NEWS_MARKER_OFFSET_FRAC = 0.04

MA_COLORS: dict[int, str] = {
    5:  "#60a5fa",  # blue
    15: "#fb923c",  # orange
    60: "#a78bfa",  # violet
}

AVG_LINE_COLORS: dict[str, str] = {
    "7d":  "#34d399",  # green
    "28d": "#fbbf24",  # amber
    "1y":  "#f472b6",  # pink
}

FIB_LEVELS: list[tuple[float, str]] = [
    (0.0,   "0%"),
    (0.236, "23.6%"),
    (0.382, "38.2%"),
    (0.5,   "50%"),
    (0.618, "61.8%"),
    (0.786, "78.6%"),
    (1.0,   "100%"),
]
