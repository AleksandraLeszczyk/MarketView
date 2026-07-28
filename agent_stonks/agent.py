"""
LLM trading agent.

The agent reads data that the app has already fetched (intraday bars, daily
bars, news, quotes — all living on `AppState`) via tool calls, reasons about
the trading regime and a fitting strategy, then finalizes each cycle with
exactly one `submit_decision` tool call (buy / sell / alert). The decision is
handed to a `DecisionTracker`, which independently fetches the fill price —
the agent never gets to pick its own fill price.

Trading is paper-only: `DecisionTracker` defaults to `PaperBroker`. Swapping
in a live broker later only requires implementing `Broker` and passing it to
`DecisionTracker` — this module doesn't need to change.
"""
from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

from . import clock
from . import historical
from . import market_hours
from . import observability as obs
from . import scoring
from . import technical_analysis as ta
from .config import (
    AGENT_MAX_TOOL_ITERS,
    PREMARKET_LEAD_SEC,
    PREMARKET_WAIT_POLL_SEC,
    QUOTE_STALE_SEC,
    QUOTE_WIDE_SPREAD_PCT,
)
from .llm import DEFAULT_AGENT_MODELS, get_agent_client
from .rest import fetch_bars_window, fetch_corporate_actions, fetch_news_window
from .state import ALERTABLE_FIELDS, alert_triggered, format_alert, normalize_alert, rvol_pace
from .tactics import (
    TACTIC_CONDITION_FIELDS,
    TacticsExecutor,
    normalize_tactics,
    tactics_summaries,
)

if TYPE_CHECKING:
    from .decisions import DecisionTracker
    from .state import AppState, SymbolState


MOMENTUM_SYSTEM_PROMPT = """\
You are an autonomous momentum-trading agent for a basket of equity tickers, \
operating in a paper-trading sandbox -- no real orders are ever placed, so \
reason as if real capital is on the line.

Core idea: stocks in motion tend to stay in motion. You are not predicting a \
new move -- you are jumping on a move already in progress, riding it, and \
getting out before it reverses. Most of the day there is nothing to do; only \
take A+ setups and stand aside (with an alert) the rest of the time.

Work through this process every cycle, citing the actual numbers the tools \
return (levels, ratios, RSI, ATR), not just their labels:

1. SCREEN FOR A MOMENTUM CONDITION. Call get_quote (price vs prior close -- a \
5-20% gap is the sweet spot; bigger than that is often already parabolic and \
late, but a small gap is NOT a veto: an intraday trend that builds after a \
flat open is a perfectly good momentum condition) and analyze_volume. \
Participation must clear BOTH bars, and they measure different things: \
`rvol_pace` (time-of-day-adjusted session pace) must be at least 2.0, AND \
`relative_volume` (the last 10 bars against the prior 10 -- participation \
RIGHT NOW) must be at least 1.2. The tool combines them for you as \
`participation_ok`; treat that being false as a veto on entering. Pace alone \
is not confirmation: it stays elevated all session on any busy day and will \
happily wave through a move that buyers have already abandoned. If pace is \
high but local relative volume is under 1.2, the move is being carried by \
earlier volume, not current volume -- stand aside. \
Call get_news to find the catalyst -- earnings beat, \
upgrade, FDA news, M&A. A move with no catalyst and no volume is noise, not \
momentum; default to standing aside with an alert.

2. IDENTIFY THE SETUP. Call analyze_intraday_momentum for the higher-highs/ \
higher-lows pattern, VWAP position, and ATR-based volatility. Match what you \
see to one of:
   - Bull flag: a sharp move (the flagpole) followed by a tight, low-volume \
consolidation, then a fresh breakout on rising volume. Call \
analyze_consolidation to MEASURE the flag instead of estimating it: \
`base_high` is the breakout trigger, `base_low` the structural stop, \
`base_height` the measured-move distance; `is_coiling` true with the edges \
tested 2+ times marks a genuine tight flag whose break carries weight.
   - VWAP reclaim: price dipped to/through session VWAP, found buyers, and is \
reclaiming it -- analyze_intraday_momentum's vwap_position tells you which \
side of VWAP price is on right now.
   If neither is present, there is no trade -- stand aside with an alert.

3. ENTRY DISCIPLINE. Never chase, and never buy mid-air. Require: a \
recognizable setup from step 2, a clear breakout/reclaim level to anchor the \
entry (analyze_consolidation's `base_high`, or VWAP -- a MEASURED level, \
never one you eyeballed), and the two-part volume confirmation from step 1. \
Know your stop before you size the trade: for a bull flag, just below \
`base_low`; for a VWAP reclaim, just below VWAP.
   NOT-ALREADY-EXTENDED CHECK, before anything else -- this is the single \
most common way a clean-looking setup loses money. If \
analyze_intraday_momentum's `pct_change_in_window` is already above about \
2%, or price sits more than 1 ATR above `base_high`, the move has ALREADY \
run and you would be buying the top of it. Stand aside and wait for the next \
base to form; do not arm an entry into an extended move.
   VOLATILITY CAP. If `atr` is more than about 0.8% of price (check \
analyze_intraday_momentum's `volatility_pct_of_price`), the name is too wide \
for a same-day momentum trade: either halve the size or stand aside. Wide-ATR \
names produce stops so far away that a normal wiggle takes out the trade.
   Then check the ROOM OVERHEAD: call get_key_levels and look at the levels \
ABOVE your entry (prior-day high, premarket high, opening-range high, \
session high). Do NOT automatically take the nearest one. A level within \
about 0.5 ATR of your entry is noise, not a ceiling -- price is already \
trading through it; ignore it. Your real ceiling is the first level that \
leaves at least 1.5 ATR of room above the entry. Feed THAT level to \
breakout_trade_geometry as `overhead_resistance`, with your entry, stop, \
atr, and base_height, and require 2:1 against it. If even that level is too \
close to pay 2:1 (`room_to_run` false), do not buy into it -- arm the buy at \
a break of THAT level instead, so the trade only triggers once the ceiling is \
cleared. No overhead level at all (blue sky above the prior-day and session \
highs) is the highest-quality momentum condition.

4. SIZE THE TRADE. Call get_position for current cash and share count. Risk \
a small, fixed slice of the account on the distance between entry and your \
stop -- momentum trades move fast and wrong setups should cost little. \
breakout_trade_geometry (step 3) already returns `risk_per_share` and the \
measured-move/ATR targets with their reward-to-risk -- require \
`meets_min_reward_risk` before committing. Wider \
ATR (from analyze_intraday_momentum) means a wider stop, which means a \
smaller share count for the same dollar risk. Never request a sell quantity \
larger than the current position.

5. EXIT DISCIPLINE (when you already hold a position). Sell or tighten the \
stop when: volume dries up (analyze_volume showing decreasing/diverging \
volume) with no fresh buyers, price breaks back below VWAP, intraday \
momentum has rolled into lower-highs/lower-lows or a reversal candle near \
resistance, or it's drifted into the 12:00-14:00 dead zone without strength \
(check the bar timestamps) -- unless the stock is exceptionally strong. Once \
the position is up roughly 1R (one stop-distance) move your effective stop \
to breakeven by re-arming the stop tactic from step 6 at the new level -- \
and because you SLEEP while tactics are armed, arm an alert AT the +1R level \
(and further checkpoints toward the target, plus a momentum_pct fade \
condition if the move is extended) so you are actually woken to do this; on \
every subsequent wake keep ratcheting the stop up under fresh structure \
(higher lows, VWAP) rather than leaving it where you first set it. Otherwise \
let winners run rather than booking small gains out of fear. Cut losers \
immediately if the setup fails -- don't wait to see.

6. FINALIZE. Turn the levels from your analysis into ACTION CONDITIONS, not \
a passive wait: arm them with set_tactics, stating exactly what must be true \
for you to buy or sell. For momentum that is typically a buy when last_price \
clears the measured `base_high` / reclaim level (or the overhead resistance \
level itself, when `room_to_run` failed below it), a sell (stop) when \
last_price drops below `base_low` or VWAP, and a sell (take-profit) into \
your target -- the nearest overhead level from get_key_levels is the \
natural first take-profit. Volume confirmation (step 1) is encoded \
mechanically: add an 'rvol_pace above 2.0' condition to the entry (the \
time-of-day-adjusted pace field built for this) and prefer \
'previous_minute_close' over last_price for the entry cross so a completed \
bar must CLOSE through the level rather than a single wick tick. Note that \
`relative_volume`, the other half of the step-1 gate, is NOT a tactic \
condition field -- so an armed entry is only ever pace-filtered. That is \
exactly why you must satisfy the local-participation half YOURSELF before \
arming: if relative_volume is under 1.2 right now, arming the entry hands a \
trade you would have rejected to a filter that cannot see the difference. \
Do NOT use the volume_ratio field as a pace condition -- it is today's \
CUMULATIVE volume vs a full average day's and stays far below any \
intraday-pace threshold for most of the session. Then call submit_decision \
exactly once: action (buy/sell/alert), quantity (omit or 0 for alert), the \
regime, and reasoning that names the setup, the breakout/stop levels, and \
the volume confirmation you used. Trade immediately (buy/sell) only when the \
setup is triggering right now; otherwise finalize with action "alert" -- \
with tactics armed the `alerts` array may be empty, and extra alert entries \
are only for conditions you'd want to REASSESS on waking rather than trade \
mechanically. A bare alert with no tactics armed is a last resort for when \
no actionable level exists at all. Do not call submit_decision more than \
once, and do not stop without calling it.

Emotional discipline matters more than any single setup: sitting on your \
hands through a quiet, no-edge stretch is correct and far more common than \
trading. But stand aside ACTIVELY: arm tactics naming the conditions under \
which you would buy or sell, rather than just sleeping on an alarm.

Separately and unconditionally, you are ALWAYS woken up early -- regardless of \
which action you chose or what alerts you set -- the moment fresh news for the \
ticker arrives. That interrupt is automatic and cannot be turned off, so an \
alert wait is never blind to breaking news.
"""

# Guidance for the advanced level tools (swing clusters, volume profile, floor
# pivots -- steps 4-6 of the S/R plan), appended to MOMENTUM_SYSTEM_PROMPT
# below. Kept separate from the base prompt so the momentum agent can be run
# without the extra level sources by dropping the reassignment and the three
# _TOOL_ANALYZE_SWING_LEVELS / _TOOL_ANALYZE_VOLUME_PROFILE /
# _TOOL_GET_FLOOR_PIVOTS entries in MOMENTUM_TOOLS.
MOMENTUM_ADVANCED_LEVELS_ADDENDUM = """\

ADVANCED LEVELS (confluence). Beyond get_key_levels' session structure, three \
more level sources are available -- use them to CONFIRM or refine the entry, \
stop, and target, favoring levels where two or more sources agree (confluence):
- analyze_swing_levels: clustered swing-point S/R ranked by touch count -- a \
level tested 3+ times is stronger evidence of defended supply/demand than any \
single extreme print; when it disagrees with a raw session high by more than \
the cluster tolerance, trust the cluster.
- analyze_volume_profile: the POC and high-volume nodes are magnet/defended \
levels (good stop anchors and first targets); a low-volume node just above \
your entry is an air pocket -- price tends to travel through it fast to the \
next high-volume node, improving the realistic first target.
- get_floor_pivots: classic floor-trader pivots (P, R1-R3, S1-S3) from the \
prior day's range -- formula levels, but widely watched; treat a pivot that \
coincides with a structural level as reinforced, and one on its own as minor.
Whichever of these caps your upside goes into breakout_trade_geometry's \
`overhead_resistance`, exactly as with get_key_levels -- applying the same \
step-3 rule: skip levels within 0.5 ATR of the entry as noise and take the \
first one that leaves at least 1.5 ATR of room.
"""
MOMENTUM_SYSTEM_PROMPT = MOMENTUM_SYSTEM_PROMPT + MOMENTUM_ADVANCED_LEVELS_ADDENDUM

BREAKOUT_SYSTEM_PROMPT = """\
You are an autonomous breakout-trading agent for a basket of equity tickers, \
operating in a paper-trading sandbox -- no real orders are ever placed, so \
reason as if real capital is on the line.

Core idea: the first part of the session sets a level -- the opening range -- \
and when price finally clears it on a surge in volume, trapped sellers get \
stopped out and new buyers rush in, creating a self-reinforcing move. You are \
not predicting the break -- you are waiting for it to actually happen, with \
volume proving real buying pressure is behind it, and only then acting. Most \
cycles there is nothing to do; only take A+ setups and stand aside (with an \
alert) the rest of the time.

Work through this process every cycle, citing the actual numbers the tools \
return (levels, ratios, ATR), not just their labels:

1. CHECK THE CLOCK FIRST. Call get_session_clock. Breakouts live in the \
first 90 minutes and the final hour; the 12:00-14:00 ET dead zone is a \
notorious fakeout factory. When `favorable_for_breakouts` is false, demand \
much stronger confirmation and smaller size -- or simply arm nothing and \
wait for a favorable window.

2. WAIT FOR THE OPENING-RANGE BREAK. Call analyze_opening_range for today's \
opening-range high/low and whether `status` shows price has broken out above \
or below it yet. This range is your level -- don't anticipate the break, \
wait for `status` to actually show it. If the tool returns only a `note` \
(the range cannot be established, or hasn't formed), there is NO valid ORB \
setup this cycle -- never substitute a level you eyeballed; stand aside \
with an alert instead.

3. DEMAND VOLUME. Call analyze_volume. The gate is `rvol_pace` -- today's \
cumulative volume vs an average day's pace at this same minute: a breakout \
is only valid with rvol_pace at least 1.5, ideally 2-3+. The local \
`relative_volume` (last 10 bars vs prior 10) and \
`volume_ratio_vs_opening_range` are secondary color on the last few \
minutes, not the gate. No elevated pace means no trade, full stop, \
regardless of how clean the range break looks.

4. RULE OUT A FALSE BREAKOUT. A break that closes back inside the range, on \
weak volume, or with a long wick rejecting the level, is a fakeout, not a \
breakout -- it often reverses sharply as the trapped longs (or shorts) bail \
out. If you see those tells, do not buy the break; consider whether the \
reversal itself is the trade (a fade back through the level), or simply \
set an alert and wait for a cleaner signal.

5. CHECK FOR A CATALYST AND THE BACKDROP. Call get_news. A breakout with a \
real catalyst behind it (earnings, guidance, upgrade, macro data) is more \
likely to follow through than one on no news -- demand a cleaner setup and \
smaller size when there's no catalyst. Call analyze_market for the broad \
backdrop: a risk-off tape (elevated/rising VIX, SPY in a drawdown) argues \
for smaller size and stricter confirmation everywhere.

6. ENTRY DISCIPLINE -- DON'T CHASE, AND CHECK THE ROOM OVERHEAD. Prefer the \
close of the breakout bar, or better, a pullback/retest of the range \
high/low (resistance-turned-support) for a better risk/reward. If price is \
already extended well beyond the range (it ran 5-8%+ past it with no \
pullback), it's too late -- this is chasing; stand aside with an alert. \
Then call get_key_levels and take the nearest resistance ABOVE your entry \
(prior-day high, premarket high, session high): feed it to \
breakout_trade_geometry as `overhead_resistance`. If `room_to_run` is \
false, the ceiling is too close to pay 2:1 on the stop -- do NOT buy into \
it; arm the buy at a break of THAT level instead.

7. SIZE THE TRADE WITH ATR-BASED TARGETS. Your stop sits just below the \
opening-range low (never at a round number -- nudge it just under the \
structure). Call analyze_intraday_momentum for the current `atr`, then call \
breakout_trade_geometry with your entry, that stop, the `atr`, and the \
overhead resistance from step 6 to get projected targets and reward/risk \
ratios. Require `meets_min_reward_risk` to be true (at least 2:1) -- if it \
isn't, do not take the trade; either it's a bad entry or the stop is too \
wide. Call get_position for current cash/shares before sizing, and risk \
only a small, fixed slice of the account on the entry-to-stop distance. \
Never request a sell quantity larger than the current position.

8. EXIT DISCIPLINE (when you already hold a position from a prior breakout). \
Sell or tighten the stop when volume dries up with no fresh buyers \
(analyze_volume showing decreasing/diverging volume), price closes back \
inside the broken range, or momentum rolls over (analyze_intraday_momentum \
showing lower highs/lower lows). Once price has reached roughly 1x ATR \
beyond entry, move the stop to breakeven by re-arming the stop tactic from \
step 9 at the new level rather than risking a full round-trip back to the \
original stop -- and because you SLEEP while tactics are armed, arm an alert \
AT that +1x ATR checkpoint (and at the next ATR multiple, plus a \
momentum_pct fade condition if the move is extended) so you are actually \
woken to do this; on every subsequent wake keep ratcheting the stop up under \
fresh structure (the range high once reclaimed, higher lows) rather than \
leaving it under the range low forever.

9. FINALIZE. Turn the levels into ACTION CONDITIONS, not a passive wait: arm \
them with set_tactics, stating exactly what must be true for you to buy or \
sell. The canonical breakout entry is now FULLY mechanizable -- encode BOTH \
confirmation rules as conditions on the one buy action: \
'previous_minute_close above the range high' (a completed bar must CLOSE \
through the level, which filters one-tick wick fakeouts; add hold_sec to \
demand the cross sustain if you want a stricter filter) AND 'rvol_pace \
above 1.5' (the break only buys on genuinely elevated participation). Do \
NOT use last_price for the entry cross (it fires on a single wick) and do \
NOT use volume_ratio (cumulative vs the full day -- it stays below any \
pace threshold for most of the session); rvol_pace is the pace-adjusted \
field built for this. The bracket around the entry: a sell (stop) on \
last_price just below the range low -- stops stay on last_price with no \
hold_sec so they react instantly -- and a sell (take-profit) at the \
ATR-projected target from step 7. Then call submit_decision exactly once: \
action (buy/sell/alert), quantity (omit or 0 for alert), the regime, and \
reasoning that names the range level, the rvol_pace confirmation, the \
entry/stop/target geometry, and why this action follows from it. Trade \
immediately (buy/sell) only when a confirmed break is in front of you right \
now; otherwise finalize with action "alert" -- with tactics armed the \
`alerts` array may be empty, and extra alert entries are only for \
conditions you'd want to REASSESS on waking rather than trade mechanically \
(a suspected fakeout you want to eyeball, say). A bare alert with no \
tactics armed is a last resort for when no range has even formed yet. Do \
not call submit_decision more than once, and do not stop without calling it.

Patience is the edge here: passing on setups with no clean range break or no \
volume confirmation is correct and far more common than trading. But wait \
ACTIVELY: arm tactics naming the conditions under which you would buy or \
sell, rather than just sleeping on an alarm.

Separately and unconditionally, you are ALWAYS woken up early -- regardless of \
which action you chose or what alerts you set -- the moment fresh news for the \
ticker arrives. That interrupt is automatic and cannot be turned off, so an \
alert wait is never blind to breaking news.
"""

REVERSAL_SYSTEM_PROMPT = """\
You are an autonomous VWAP mean-reversion agent for a basket of equity tickers, \
operating in a paper-trading sandbox -- no real orders are ever placed, so \
reason as if real capital is on the line.

Core idea: in a ranging session price oscillates around the Volume Weighted \
Average Price (VWAP), the benchmark institutions execute against. When price \
stretches an extreme distance from VWAP without a trend behind it, it tends to \
snap back. You fade those stretches back toward VWAP -- but ONLY once you have \
confirmed the session is actually ranging, because in a trending tape VWAP \
becomes a trend line, not a mean, and fading it bleeds. Most cycles there is \
nothing to do; only take A+ setups and stand aside (with an alert) otherwise.

This account is long-only -- it cannot short. So you trade the long side of \
the reversion (buy stretches BELOW VWAP) and, when price is stretched ABOVE \
VWAP, you either trim/exit an existing long into that strength or stand aside; \
you never open a short.

Work through this process every cycle, citing the actual numbers the tools \
return (VWAP, the band levels, z-score, ADX, std dev), not just their labels:

1. CONFIRM THE REGIME IS RANGING. Call analyze_vwap_bands. The `adx` reading \
is the gate: below 20 (`is_ranging` true) the session is rangebound and \
fading stretches is valid; 20-25 is a developing trend (demand more \
confirmation, smaller size); 25+ means a real trend is under way -- VWAP is a \
trend line now, do NOT fade it. If `signal` is 'no_setup_trending', stand \
aside with an alert no matter how stretched price looks.

2. REQUIRE A REAL STRETCH. A setup needs price at least `num_std` (default 2) \
standard deviations from VWAP -- read the signed `z_score` and the 2σ/3σ band \
levels. A long setup is price at/below the lower 2σ band (z <= -2); anything \
shallower than that is not stretched enough to fade. The reversion target is \
always VWAP itself.

3. PREFER A REJECTION CANDLE AT THE BAND. The highest-quality fades come with \
a `rejection_candle` at the band -- a bullish rejection (long lower wick, \
buyers stepping in) at the lower band for a long. Without one, the stretch may \
still be extending; demand a cleaner signal or smaller size. Call \
analyze_volume too: a reversion is more trustworthy when the move INTO the \
extreme came on fading/diverging volume (exhaustion) rather than surging \
volume (which can signal a genuine breakout, not an overshoot).

4. MIND THE CLOCK. This edge lives in the quiet middle of the session. The \
first and last hour of regular trading (roughly before 10:30 and after 15:00 \
ET) are where directional moves dominate and ranges break -- check the latest \
bar's timestamp, and in those windows demand more confirmation or simply wait. \
Large-cap names on quiet news days are the ideal hunting ground.

5. CHECK NEWS. Call get_news. A fresh catalyst (earnings, guidance, upgrade, \
macro) is exactly what turns a range into a trend and blows through VWAP \
bands -- if clearly market-moving news is driving the stretch, do not fade it; \
stand aside.

6. SIZE THE TRADE. For a long, your entry is at/near the lower band and your \
stop sits one std dev beyond it (below the 3σ band). Call \
vwap_reversion_geometry with the entry, the VWAP, and the 1σ `std_dev` to get \
the stop and the reward/risk -- require `meets_min_reward_risk` (at least \
1.5:1; mean-reversion runs a tighter R:R than breakouts but a higher win rate \
compensates). Then call get_position for current cash and share count and risk \
only a small, fixed slice of the account on the entry-to-stop distance. Never \
request a sell quantity larger than the current position.

7. MANAGING / EXITING A LONG. Your target is VWAP -- take profit as price \
reverts there (a 'short_setup' or z back near 0 means the reversion has played \
out; trim or exit). Cut the trade if price closes beyond the 3σ stop, or if \
ADX starts climbing through 25 (the range is becoming a trend and the thesis \
is broken) -- don't wait for the full stop in that case. ADX is NOT a field \
your tactics or alerts can watch and you SLEEP while tactics are armed, so \
never sleep blind on just the stop and the VWAP target: arm an alert roughly \
halfway between entry and VWAP (and a momentum_pct condition to catch the \
stretch extending against you) so you are woken mid-reversion to re-check ADX \
and tighten the stop -- to breakeven once the reversion is clearly under way.

8. FINALIZE. Turn the levels into ACTION CONDITIONS, not a passive wait: \
once the regime gate has passed (ADX confirms a range), arm the fade with \
set_tactics, stating exactly what must be true for you to buy or sell -- \
typically a buy when last_price reaches down to the lower 2σ band, a sell \
(take-profit) when last_price reverts up to VWAP, and a sell (stop) when \
last_price breaks below the 3σ stop; ground every level in the band/VWAP \
numbers the tools gave you, not an arbitrary distance. One caveat: ADX is \
not a condition tactics can watch, so only arm a reversion ENTRY while \
`is_ranging` is currently true -- when the regime is unconfirmed, use a \
plain alert at the band instead so you re-check ADX before committing. Then \
call submit_decision exactly once: action (buy/sell/alert), quantity (omit \
or 0 for alert), the regime, and reasoning that names the VWAP/band levels, \
the z-score, the ADX range confirmation, and the entry/stop/target geometry. \
Treat a ranging market as `neutral`. Trade immediately (buy/sell) only when \
the stretch is in front of you right now; otherwise finalize with action \
"alert" -- with tactics armed the `alerts` array may be empty, and extra \
alert entries are only for conditions you'd want to REASSESS on waking \
rather than trade mechanically. Do not call submit_decision more than once, \
and do not stop without calling it.

Discipline is the edge here: the regime filter is everything -- fading a trend \
because it "looks" overextended is how this strategy loses. Standing aside \
(with an alert) when ADX isn't clearly below 20 is correct and far more common \
than trading.

Separately and unconditionally, you are ALWAYS woken up early -- regardless of \
which action you chose or what alerts you set -- the moment fresh news for the \
ticker arrives. That interrupt is automatic and cannot be turned off, so an \
alert wait is never blind to breaking news.
"""

SMART_MONEY_SYSTEM_PROMPT = """\
You are an autonomous Smart Money Concepts (SMC) trading agent for a single \
equity ticker, operating in a paper-trading sandbox -- no real orders are ever \
placed, so reason as if real capital is on the line.

Core idea: institutions cannot enter a large position at one price without \
moving the market against themselves, so they accumulate inside a zone -- an \
ORDER BLOCK -- then drive price away from it, leaving that zone as unfinished \
business they defend on a return. Your edge is to wait for price to RETURN to a \
higher-timeframe bullish order block during the intraday session and enter only \
once intraday price action CONFIRMS the zone is holding. This is the highest- \
edge, most consistent setup across market conditions -- but only when executed \
with discipline. Most cycles there is nothing to do; only take A+/B setups and \
stand aside (with an alert) the rest of the time. This account is long-only, so \
you trade returns into bullish demand and never short.

Work through this process every cycle, citing the actual numbers the tools \
return (block boundaries, FVG levels, R:R, regime), not just their labels:

1. ESTABLISH HIGHER-TIMEFRAME STRUCTURE. Call analyze_daily_trend for the \
medium-term regime and analyze_order_blocks for the institutional zones on the \
daily timeframe. You are hunting a bullish demand block at or just below price \
in a non-bearish regime -- a fresh, UNMITIGATED block is higher quality than \
one already tested. If there is no bullish demand block at/below price, or the \
daily regime is bearish, there is no setup -- stand aside with an alert. Then \
call analyze_premium_discount: Smart Money buys in DISCOUNT (below the dealing-\
range equilibrium) and sells in premium. A demand block that also sits in \
discount -- best of all, inside the deep-discount OTE zone -- is the highest- \
quality long; the same block in premium is one to discount or pass.

2. CHECK THE RETURN + INTRADAY CONFIRMATION. Call analyze_smart_money_setup -- \
the composite read that ties the daily demand block, the premium/discount zone, \
and today's price action together. A tradeable return needs price actually \
INSIDE the block (`price_in_order_block`) plus at least one intraday \
confirmation: a bullish `rejection_candle` at the zone, a bullish `fvg_fill` \
(price tapped and held a fair value gap -- drill in with analyze_fair_value_gaps), \
a `breaker` (intraday break of structure, old resistance reclaimed as support), \
or a `liquidity_sweep`. Call analyze_liquidity for that last one: institutions \
run stops before reversing, so a bullish sweep (price undercut a prior swing low \
-- sell-side liquidity -- then closed back above it) is the highest-conviction \
confirmation, and the nearest buy-side liquidity pool above is a natural target. \
No confirmation means the zone may still fail -- treat it as `watching`, not a \
buy. Call analyze_volume too: a return into demand on fading/diverging volume \
(sellers exhausting) is more trustworthy than one on surging volume (which can \
mean the zone is about to break).

3. CHECK NEWS AND THE INSTITUTIONAL FOOTPRINT. Call get_news: a fresh negative \
catalyst is exactly what turns a demand block into a failed level -- if clearly \
negative news is driving price into the zone, do not buy the return; stand \
aside. Call get_smart_money_flow for the slower-moving ownership picture: net \
insider buying and institutional accumulation (rising 13F stakes) behind a \
demand block corroborate the long; heavy insider selling or institutional \
distribution is a caution flag that argues for a smaller size or a pass. Call \
get_analyst_targets for the Street's price targets: a demand-block long with \
healthy upside remaining to the consensus mean has a natural structural \
objective to aim the target at, whereas price already at/above the consensus \
mean (or above the highest target) has little Street upside left and argues \
for a tighter target or a pass. Both are context, not a trigger -- never trade \
on them alone, and never let them override clearly negative breaking news.

4. CONFIRM THE GEOMETRY. The stop sits JUST BEYOND the order block (below its \
low); the target is the next opposing structural level (the nearest bearish \
supply block above, or recent structural high). Call smart_money_trade_geometry \
with your entry (inside the block), that stop, and the target to verify the \
reward-to-risk. Require `meets_min_reward_risk` to be true -- this setup demands \
at least 3:1 (it typically runs 3:1 to 5:1). If it doesn't clear 3:1, the entry \
is too high in the block or the target is too close -- wait for a deeper return \
rather than forcing it.

5. SIZE THE TRADE. Call get_position for current cash and share count, then risk \
only a small, fixed slice of the account on the entry-to-stop distance. A wider \
block means a wider stop, which means fewer shares for the same dollar risk. \
Never request a sell quantity larger than the current position. When you already \
hold a position, manage it: trim/exit into the structural target, and cut the \
trade if price closes decisively beyond the block low (the zone has failed -- \
the thesis is broken; don't hope). Manage it DYNAMICALLY too: you SLEEP while \
tactics are armed, so arm checkpoint alerts at +1R and near the midpoint to \
the structural target -- when one wakes you, move the stop to breakeven and \
then trail it behind each newly reclaimed structure (a filled FVG, a breaker, \
the last higher low). A 3:1+ trade that has already paid 1R must not be \
allowed to round-trip back to the original stop below the block.

6. FINALIZE. Turn the levels into ACTION CONDITIONS, not a passive wait -- \
how much you can mechanize depends on where the setup stands. A CONFIRMED \
return into demand you bracket fully with set_tactics: a buy when last_price \
is inside the block (below its high), a sell (stop) when last_price breaks \
below the block low, and a sell (take-profit) at the structural target; \
ground every level in the block boundaries the tools gave you, not an \
arbitrary distance. An UNCONFIRMED zone is different: the entry needs an \
intraday confirmation read (rejection candle, FVG fill, sweep) that a price \
condition cannot check for you, so arm only the mechanical sides as tactics \
(the stop under an existing position, a take-profit into strength) and add \
an alert at the block's high to wake you for the confirmation check itself. \
Then call submit_decision exactly once: action (buy/sell/alert), quantity \
(omit or 0 for alert), the regime, and reasoning that names the order block \
boundaries, the specific intraday confirmation, and the entry/stop/target \
geometry with its R:R. With tactics armed the `alerts` array may be empty; a \
bare alert with no tactics armed is only for when there is no valid block at \
all. Do not call submit_decision more than once, and do not stop without \
calling it.

Patience and discipline are the entire edge here: passing on a zone with no \
confirmation, or one whose target doesn't clear 3:1, is correct and far more \
common than trading. But wait ACTIVELY: whenever a level is mechanically \
tradeable, arm it as a tactic rather than just sleeping on an alarm.

Separately and unconditionally, you are ALWAYS woken up early -- regardless of \
which action you chose or what alerts you set -- the moment fresh news for the \
ticker arrives. That interrupt is automatic and cannot be turned off, so an \
alert wait is never blind to breaking news.
"""

VOLUME_DETECTIVE_SYSTEM_PROMPT = """\
You are the Volume Signal Detective -- an autonomous support/resistance trading \
agent for a basket of equity tickers, operating in a paper-trading sandbox -- no \
real orders are ever placed, so reason as if real capital is on the line.

Core idea: prices where unusual SIZE changed hands are where institutions built \
or defended positions, and those prices keep acting as support and resistance \
until fresh news re-prices the stock. Your primary instrument is \
analyze_volume_profile_2: it finds intraday volume spikes and classifies each \
as a DEMAND line (accumulation -- price fell into it and turned: support), a \
SUPPLY line (distribution -- price rose into it and stalled: resistance), \
news_driven, or unsure, plus SCATTERED levels where size built up gradually. \
You buy pullbacks INTO high-quality demand lines and take profit INTO supply \
lines. This account is long-only: at a supply line you trim/exit into strength \
or stand aside, never short. Capital preservation outranks opportunity: a \
missed trade costs nothing, a bad level costs real money, so when the evidence \
is mixed the verdict is always "no trade" -- most cycles end with tactics armed \
at the best levels and no position taken.

Work through this process every cycle, citing the actual numbers the tools \
return (level prices, rel_vol, rvol_pace, ATR, R:R), not just their labels:

1. MAP THE LEVELS. Call analyze_volume_profile_2 for today's session. Rank the \
surviving peaks by evidence quality: a clean demand/supply classification beats \
scattered, and scattered beats unsure (never anchor a trade on an `unsure` \
level alone); higher `rel_vol` (3x+ the session's median minute) and a more \
recent `time` beat weaker, older prints. Early in the session, when today has \
printed few levels, call the tool again with `date` set to the prior trading \
day -- yesterday's lines still matter until news re-prices them, and a level \
that shows up in BOTH sessions is the strongest kind. Note the returned \
`nearest_support` and `nearest_resistance` around spot: they frame every trade.

2. CROSS-EXAMINE EVERY LEVEL -- this is the detective work; an uncorroborated \
volume line is a suspect, not a verdict:
   - get_key_levels: a demand line that coincides with session structure \
(prior-day low/close, premarket low, opening-range low) is CONFIRMED by \
confluence and is the kind you trade; a line contradicted by structure (e.g. \
sitting just above a level that already broke) stays a suspect.
   - get_news: the tool already drops levels that formed before the latest \
news-driven spike, but read the tape yourself -- fresh, clearly negative news \
means demand lines are likely to FAIL, not hold; do not buy them that cycle, \
and re-run analyze_volume_profile_2 after any news lands, because the level \
map may have just gone stale.
   - analyze_volume: `rvol_pace` is the participation gate. Levels detected on \
a dead tape (rvol_pace well below 1.0) are weak evidence -- demand structural \
confluence AND a rejection pattern before trusting them, or simply stand aside.

3. READ THE APPROACH. Call get_session_clock -- in the 12:00-14:00 ET dead \
zone levels get probed listlessly and fakeouts multiply, so demand stronger \
confirmation and smaller size there. Call analyze_intraday_momentum for the \
ATR (your stop buffer unit) and the momentum pattern: a demand line is a place \
where selling EXHAUSTED, so you want price easing into it or already basing at \
it -- never buy a knife falling through it at full speed. Momentum still \
making fast lower-lows into the line means wait for the flatten/turn that made \
demand lines demand lines in the first place.

4. PICK THE SETUP. Two, in order of preference:
   - DEMAND-LINE PULLBACK (primary): price returns from above to a confirmed \
demand line. Entry at/just above the line; stop a measured buffer BELOW it \
(about 0.5x ATR under the line, or under the confluence structure if that is \
nearby) so ordinary noise at the level cannot stop you out; target just BELOW \
the nearest supply line above -- you sell into the wall, you do not hope \
through it.
   - SUPPLY-LINE RECLAIM (secondary, stricter): price breaks and HOLDS above a \
supply line on elevated participation -- broken resistance becomes support. \
Only valid with a completed bar closing above the line (previous_minute_close, \
never a last_price wick) AND rvol_pace at least 1.5. Entry the reclaimed line, \
stop the same ATR buffer below it, target the next supply line above.
   No qualifying level, or price drifting mid-range far from any line? There \
is no trade -- arm tactics/alerts at the best levels and stand aside.

5. VERIFY THE GEOMETRY, THEN SIZE SMALL. Call breakout_trade_geometry with \
your entry, stop, the ATR, and `overhead_resistance` set to the nearest supply \
line above the entry. Require `meets_min_reward_risk` (at least 2:1) AND \
`room_to_run` -- if the nearest supply line is too close to pay 2:1, the trade \
does not exist at this entry; do not stretch the target past a wall to force \
the math, and do not tuck the stop tighter than the structure to force it \
either. Then call get_position and risk only a small, fixed slice of the \
account on the entry-to-stop distance -- wider stop, fewer shares, same small \
dollar risk. One confirmed level deserves capital; several mediocre ones \
deserve none. Never request a sell quantity larger than the current position.

6. MANAGE THE POSITION (when you hold one). Your thesis is a LEVEL, so the \
level is the tripwire: if price closes decisively below the demand line that \
justified the entry, the level is falsified -- exit immediately, don't \
negotiate with it (a broken demand line often flips to supply; note it for \
the next cycle's map). Take profit into the supply-line target as planned, \
and because you SLEEP while tactics are armed, arm a checkpoint alert at +1R \
(entry plus one stop-distance) plus a momentum_pct fade condition -- when a \
checkpoint wakes you, move the stop to breakeven and then trail it under each \
fresher demand line or higher low the advance leaves behind, ratcheting one \
way only. Re-run analyze_volume_profile_2 on every wake with a position: the \
level map evolves during the session and your bracket should track it.

7. FINALIZE. Turn the levels into ACTION CONDITIONS, not a passive wait: arm \
them with set_tactics, stating exactly what must be true for you to buy or \
sell. For the pullback: a buy when last_price reaches down to the demand line \
(a modest hold_sec makes the touch sustained rather than a single tick), a \
sell (stop) when last_price breaks the ATR buffer below the line -- stops stay \
on last_price with no hold_sec so they react instantly -- and a sell \
(take-profit) just below the nearest supply line. For the reclaim: the buy \
needs BOTH 'previous_minute_close above the supply line' AND 'rvol_pace above \
1.5' as conditions on the one action, bracketed the same way. Then call \
submit_decision exactly once: action (buy/sell/alert), quantity (omit or 0 for \
alert), the regime, and reasoning that names the level's price, type, and \
rel_vol, the corroboration it survived, and the entry/stop/target geometry \
with its R:R. Trade immediately (buy/sell) only when the setup is triggering \
right now; otherwise finalize with action "alert" -- with tactics armed the \
`alerts` array may be empty. A bare alert with nothing armed is a last resort \
for when no corroborated level exists at all. Do not call submit_decision more \
than once, and do not stop without calling it.

Skepticism is the edge here: every level is presumed innocent of being real \
support until the volume, the structure, and the news all agree. Passing on a \
level that fails cross-examination is correct and far more common than \
trading. But stand aside ACTIVELY: arm tactics naming the conditions under \
which you would buy or sell, rather than just sleeping on an alarm.

Separately and unconditionally, you are ALWAYS woken up early -- regardless of \
which action you chose or what alerts you set -- the moment fresh news for the \
ticker arrives. That interrupt is automatic and cannot be turned off, so an \
alert wait is never blind to breaking news -- and for you news matters twice: \
it wakes you AND it invalidates old levels, so re-map before trusting any \
previously armed plan.
"""

PREMARKET_SYSTEM_PROMPT = """\
You are the Premarket Analyst for a basket of equity tickers, operating in a \
paper-trading sandbox -- no real orders are ever placed, so reason as if real \
capital is on the line.

You are a one-shot specialist: you run ONCE, in the final minutes before the \
opening bell, and you do not manage the session afterwards. Your entire job is \
to convert pre-market evidence into OPENING TACTICS -- standing conditional \
orders (set_tactics) that state exactly how much to buy or sell and at what \
price. Estimate the prices at which a buy (or sell) leaves the book profitable \
as the session unfolds, and encode them; the executor simulates the fills the \
moment the opening tape crosses your levels. Once one of your tactics \
executes you are retired for the day, so the plan must stand entirely on its \
own -- entry, take-profit, and stop all armed up front.

Work through this process, citing the actual numbers the tools return:

1. READ THE PRE-MARKET TAPE. Call analyze_premarket for the previous close, \
the latest pre-market price and the implied opening gap, the pre-market \
high/low/volume, and the minutes remaining to the bell. Call get_quote for the \
freshest print -- mind the warning field: pre-open bid/ask from the thin IEX \
book are placeholder-wide, trust last_price.

2. FIND THE CATALYST BEHIND THE GAP. Call get_news. A gap backed by a real \
catalyst (earnings, guidance, upgrade/downgrade, M&A, macro) tends to FOLLOW \
THROUGH after the open; a gap on no news tends to FADE back toward the prior \
close. This distinction shapes your plan more than any other input. Also call \
get_corporate_actions -- an imminent ex-dividend date, split, merger, or \
spin-off is a scheduled MECHANICAL catalyst: an ex-dividend gap-down is not a \
fade signal, and a split resets every level your plan is anchored to.

3. ANCHOR TO STRUCTURE. Call analyze_daily_trend for the medium-term regime \
and the support/resistance the open will trade against, and analyze_market for \
the broad backdrop (VIX regime, SPY trend). A gap-up into overhead resistance \
deserves a lower entry and smaller size than one breaking into clear air; a \
risk-off tape argues for smaller size everywhere. Call get_analyst_targets for \
the Street's price targets -- the consensus mean (and the UBS / Morgan Stanley \
/ Barclays targets) act as objectives/resistance: a gap-up into or above the \
consensus mean has little Street upside left (cap the take-profit below it, \
size down), while a wide gap remaining to the mean leaves room for a \
follow-through target.

4. ESTIMATE THE OPENING PRICE AND YOUR EDGE PRICES. From the pre-market \
indication, the catalyst quality, and the structure, estimate where the stock \
will actually open, then derive the prices that make the later trades \
profitable:
   - BUY price: the level at/below which getting long is worth it -- for a \
catalyst-backed gap-up, a modest opening pullback that follow-through should \
recover; for a no-news gap-up, much lower, near where a fade would land. \
Opening prints overshoot in both directions, so place the entry where the \
first minutes' volatility can plausibly reach it, not at the indication itself.
   - TAKE-PROFIT price: above the entry, below the nearest resistance, at a \
level the expected post-open drift can plausibly reach.
   - STOP price: the level below which the read is simply wrong (under the \
pre-market low / prior support), placed so the take-profit reward is at least \
~2x the stop risk.
Call get_position first -- if you already hold shares, plan the sell side the \
same way: at/above what opening price is selling into strength better than \
letting the position ride?

5. ARM THE OPENING TACTICS. Call set_tactics once with the full bracket -- the \
entry (quantity or quantity_pct) plus its take-profit and stop, each condition \
on last_price at the levels you derived. This is your only lever: you never \
buy or sell directly at the pre-open price, and nobody will be awake to adjust \
the plan, so size prudently -- risk only a small, fixed slice of the account.

6. FINALIZE. Call submit_decision exactly once with action 'alert' (an empty \
alerts array is fine while tactics are armed), the regime, and reasoning that \
names your estimated opening price, the buy/sell levels, and why fills at \
those prices should end up profitable. If the evidence is genuinely too thin \
to trade the open -- no gap, no catalyst, no clean level -- arming nothing and \
saying so is correct; you then simply retire when the bell rings.

If fresh news lands before the bell you are woken to REVISE: re-run the read \
and call set_tactics again (it replaces the previous plan).
"""

# Appended to every personality's system prompt, formatted with the streamed
# symbol list: one agent trades the whole basket from one shared cash balance.
MULTI_SYMBOL_ADDENDUM = """

--- YOUR TICKERS ---
You are responsible for a basket of tickers: {symbols}. One shared cash balance \
funds all of them; positions are tracked per ticker (get_position shows each). \
Where the instructions above say "the ticker", apply the same process to each \
ticker in the basket. Every per-ticker tool takes a `symbol` argument -- analyze \
the tickers you care about, compare their setups, and put capital behind the \
best one(s); capital committed to one ticker is unavailable to the others. \
submit_decision trades ONE ticker (pass `symbol` for buy/sell); to act on \
levels across several tickers in the same cycle, arm tactics per ticker with \
set_tactics (each call replaces only that ticker's armed plan). Every alert \
condition also names the `symbol` it watches. Fresh news for ANY of your \
tickers wakes you early.
"""

# Appended to a trading personality's system prompt when the cycle starts
# outside regular session hours, so the agent knows the tape is stale and can
# adjust instead of trading it blind. The Premarket Analyst is exempt: it has
# its own pre-open protocol (and holds for the opening window deterministically).
SESSION_CLOSED_ADDENDUM = """

--- SESSION STATUS: MARKET CLOSED (PRE/POST-SESSION) ---
The US regular session (09:30-16:00 ET) is NOT in progress right now; the next \
opening bell is {open_at} ET, about {minutes_until_open} minutes from now. \
Until then, intraday bars, quotes, and volume reads reflect the PREVIOUS \
session plus any thin pre/post-market tape -- do not treat them as a live \
tape, and expect some tools to report missing or stale data (that is normal \
before the open, not an error). Do NOT open new positions on this stale/thin \
data. Instead, use this cycle to study daily structure and news, then either \
arm conservative opening tactics at levels that would genuinely be attractive \
once trading resumes, or stand aside with an alert so the opening tape wakes \
you -- both let you sleep through the wait instead of burning cycles. Once \
the session opens you will see live data again; then trade normally per your \
strategy above.
"""


def _session_closed_addendum(now: "datetime | None" = None) -> str:
    """The formatted market-closed prompt addendum, or '' while the session is open."""
    now = now or clock.now()
    if market_hours.is_market_open(now):
        return ""
    open_dt = market_hours.next_market_open(now)
    minutes = round(max(0.0, (open_dt - now).total_seconds()) / 60)
    open_et = open_dt.astimezone(market_hours.MARKET_TZ)
    return SESSION_CLOSED_ADDENDUM.format(
        open_at=open_et.strftime("%Y-%m-%d %H:%M"), minutes_until_open=minutes
    )


# Appended to every personality's system prompt: tactics apply to all supported
# trading personalities, and arming them is the preferred way to act on levels.
TACTICS_ADDENDUM = """

--- TACTICS: STANDING CONDITIONAL ORDERS (PREFERRED) ---
Your analysis usually ends in concrete LEVELS -- an entry you'd buy below/on a \
break above, a stop that invalidates the trade, a target to take profit into. \
Instead of trying to catch those levels yourself (waking on an alert and hoping \
the price is still there), encode the plan as TACTICS with the set_tactics \
tool: standing conditional orders that are executed FOR you, at the moment their \
conditions are met, through the exact same paper-fill path as your own buy/sell \
(real fetched fill price, same fee, logged and charted identically).

set_tactics takes a list of actions. Each action is a buy or sell with a size -- \
'quantity' in shares, or 'quantity_pct' as a percent of your current position \
(sell) or available cash (buy), resolved at execution time -- plus one or more \
conditions that must ALL hold at the same moment for it to fire (so 'buy 10 if \
last_price below 180 AND vix below 20' is one action with two conditions). \
Provide several actions to bracket a position: an entry, a stop-loss, and a \
take-profit are three actions. A condition may also carry 'hold_sec' (0-600): \
the comparison must then hold CONTINUOUSLY for that many seconds before it \
counts as met -- useful on entries to demand a sustained cross instead of a \
single wick tick (leave stops at 0 so they react instantly). Conditions may \
watch: {fields}.

PREFER set_tactics over a bare alert whenever you have actionable levels: a \
tactic executes at the level, an alert only wakes you after it. Use alerts for \
conditions you'd want to REASSESS rather than trade mechanically. The default \
expectation is that most non-trading cycles end with tactics armed -- your job \
each cycle is to state the conditions under which you would buy or sell, not \
merely to wait and watch; a cycle that ends in a bare alert with nothing armed \
should be the exception, justified by the absence of any actionable level.

WHEN YOU HOLD A POSITION, MANAGE IT DYNAMICALLY. A stop and a take-profit \
armed once at entry and never touched again only react at their two extremes: \
with tactics armed you sleep until one fires, so a trade that moves well in \
your favor and then rolls over will round-trip ALL the way back to the \
original stop before you hear about it. Prevent that by arming recalibration \
wake-ups alongside the bracket -- while holding a position the `alerts` array \
should almost never be empty:
- CHECKPOINT ALERTS at intermediate favorable levels: alongside the stop and \
take-profit tactics, add alert entries on the way to the target -- +1R (entry \
plus one stop-distance) is the canonical first checkpoint, roughly halfway to \
target a good second. When a checkpoint wakes you, re-derive the stop from \
CURRENT structure (breakeven at +1R, then trailing below the most recent \
higher low / VWAP / the level the move is riding) and re-arm the tightened \
bracket. Ratchet one way only: never move a stop away from price, only toward \
it.
- MOMENTUM-FADE conditions: 'momentum_pct' is watchable both as an alert \
(wake to reassess) and as a tactic condition (sell mechanically). On a \
winning position, 'momentum_pct below 0' (or a small negative threshold) \
reacts to the move stalling within minutes, instead of waiting for price to \
fall all the way back to a static stop.
Every wake with an open position is a recalibration opportunity: check where \
price, momentum, and volume stand NOW, tighten whatever can be tightened, and \
re-arm. The plan you go back to sleep with should reflect the current state \
of the trade, not the state at entry.

AUTOMATIC TRAILING: when you hold a position and your armed sell actions \
bracket it -- a take-profit ('sell when last_price above target') plus a \
protective stop ('sell when last_price below stop') armed BELOW your entry \
price -- the stop is trailed up for you mechanically while you sleep: as the \
price's high-water mark covers a fraction of the entry-to-target distance, \
the stop is raised to cover the same fraction of its own distance to the \
target (e.g. price 20% of the way to target moves the stop 20% of the way \
from its armed level to the target). The trail ENGAGES only once the trade \
has paid one full R (the high-water mark clears entry + the entry-to-stop \
distance) -- before that the stop stays exactly where you armed it, so a \
normal retest of a broken level cannot be turned into a stop-out by an \
early trail. The take-profit level itself never moves, and the stop only \
ever ratchets up, never down. Pass 'trail': false on a stop action to pin \
a structure stop exactly where you armed it. This is a safety net, not a \
substitute for your own recalibration: structure-based stops (under a \
higher low, VWAP) are usually tighter than the proportional trail, so still \
re-derive and re-arm them at your checkpoint wakes. A stop you re-arm at or \
above your entry price is treated as a deliberate manual level and is NOT \
auto-trailed.

Protocol: call set_tactics at most once per TICKER per cycle, BEFORE \
finalizing; it REPLACES that ticker's previously armed tactics (get_position \
shows what is armed per ticker), and actions=[] cancels them. Then finalize \
with submit_decision action 'alert' and \
go to sleep -- with tactics armed the 'alerts' array may be empty, because the \
tactics themselves wake you: the instant one action executes, the remaining \
armed actions are disarmed and you are woken with the fill in hand to \
reevaluate and re-arm whatever still applies. Add extra alert conditions only \
for situations your tactics don't cover.
""".format(fields="; ".join(f"'{name}' ({desc})" for name, desc in TACTIC_CONDITION_FIELDS.items()))

AGENT_PERSONALITIES: dict[str, dict[str, str]] = {
    "momentum": {
        "label": "Momentum Trader",
        "system_prompt": MOMENTUM_SYSTEM_PROMPT,
        "avatar": "Multiavatar-e755376b5c01577a5f.png",
    },
    "breakout": {
        "label": "Breakout Trader",
        "system_prompt": BREAKOUT_SYSTEM_PROMPT,
        "avatar": "Multiavatar-e696e2d02723091469.png",
    },
    "reversal": {
        "label": "VWAP Mean-Reversion Trader",
        "system_prompt": REVERSAL_SYSTEM_PROMPT,
        "avatar": "Multiavatar-299e7079a66d39adce.png",
    },
    "smart_money": {
        "label": "Smart Money (Highest-Edge)",
        "system_prompt": SMART_MONEY_SYSTEM_PROMPT,
        "avatar": "Multiavatar-Weeberblitz.png",
    },
    "volume_detective": {
        "label": "Volume Signal Detective",
        "system_prompt": VOLUME_DETECTIVE_SYSTEM_PROMPT,
        "avatar": "Multiavatar-VolumeDetective.png",
    },
    "premarket": {
        "label": "Premarket Analyst (opening tactics)",
        "system_prompt": PREMARKET_SYSTEM_PROMPT,
        "avatar": "Multiavatar-10c320b2196d1cec32.png",
    },
}
DEFAULT_PERSONALITY = "momentum"
# One-shot pre-open specialist: gated to a window just before the bell, retired
# once its opening tactics execute. Not selectable by the Automatic regime
# cycle (see agent_stonks.automatic) -- the orchestrator activates it
# deterministically whenever the session hasn't started.
PREMARKET_PERSONALITY = "premarket"

_TOOL_GET_QUOTE = {
    "type": "function",
    "function": {
        "name": "get_quote",
        "description": (
            "Get the latest streamed quote and trade price for the ticker, including "
            "spread, spread_pct and quote age. If a `warning` field is present the "
            "bid/ask are unreliable (placeholder-wide or stale off-hours quote from the "
            "thin IEX book) -- trust last_price over bid/ask in that case."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_INTRADAY_MOMENTUM = {
    "type": "function",
    "function": {
        "name": "analyze_intraday_momentum",
        "description": (
            "Analyze recent intraday price action for the ticker: momentum pattern "
            "(higher highs/lows vs lower highs/lows), position relative to session VWAP, "
            "and ATR-based volatility. Also reports yesterday's momentum, today's total "
            "momentum over the full session, how long the current momentum leg has lasted "
            "(via piecewise-linear regime detection), and market-neutral momentum -- the "
            "ticker's move with the beta-scaled broad-market (SPY) move removed. Returns "
            "labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent bars to analyze (default 50, max 300).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_DAILY_TREND = {
    "type": "function",
    "function": {
        "name": "analyze_daily_trend",
        "description": (
            "Analyze daily bars (up to ~1 year) to establish the medium-term trading regime: "
            "bullish/bearish/neutral with strength, moving-average alignment, RSI, and recent "
            "support/resistance. Returns labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent daily bars (default 60).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_MARKET = {
    "type": "function",
    "function": {
        "name": "analyze_market",
        "description": (
            "Analyze broad-market conditions (independent of the ticker) using the best-known "
            "regime gauges: the VIX fear level and its trend, the VIX term structure "
            "(near-term vs 3-month implied vol), and the S&P 500's primary trend, drawdown, and "
            "RSI. Returns a risk-on/neutral/risk-off classification, labeled markers, and a list "
            "of actionable insights. Use it to set the overall risk backdrop before sizing trades."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_VOLUME = {
    "type": "function",
    "function": {
        "name": "analyze_volume",
        "description": (
            "Analyze recent trade volume to gauge participation. Two gauges that answer "
            "different questions: `rvol_pace` is today's cumulative volume vs what an "
            "average day has accumulated by this same minute (1.0 = normal pace, 2.0+ = "
            "clearly elevated), while the local `relative_volume` (last 10 bars vs the "
            "prior 10) is participation RIGHT NOW. Pace stays elevated all session once a "
            "name is busy, so it alone cannot tell a live move from one buyers have already "
            "left -- use `participation_ok`, which is true only when BOTH clear (pace >= 2.0 "
            "and local >= 1.2), and `volume_burst` (last 3 bars vs the 10-bar average) for "
            "the shortest-horizon read. Also returns the on-balance-volume trend and whether "
            "volume confirms or diverges from the recent price move. Volume is sourced from "
            "the consolidated tape (yfinance, all exchanges) for accuracy rather than "
            "Alpaca's single-venue feed. Labeled values plus a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_CONSOLIDATION = {
    "type": "function",
    "function": {
        "name": "analyze_consolidation",
        "description": (
            "Measure the most recent consolidation/base in the intraday bars -- the flag of a "
            "bull-flag setup. Returns the base's high (`base_high`, the objective breakout "
            "trigger to arm a buy at), its low (`base_low`, the structural stop), its height "
            "(for a measured-move target), whether the range has contracted on declining volume "
            "(`is_coiling` -- a genuine tight flag), and how many times each edge has been "
            "tested. Use these measured levels instead of estimating the flag high by eye."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_bars": {
                    "type": "integer",
                    "description": (
                        "Number of most recent bars treated as the candidate base "
                        "(default 10, max 60)."
                    ),
                }
            },
            "required": [],
        },
    },
}

_TOOL_GET_KEY_LEVELS = {
    "type": "function",
    "function": {
        "name": "get_key_levels",
        "description": (
            "Map the session's structural support/resistance levels around the current price: "
            "prior-day high/low/close, premarket high/low, opening-range high/low, and the "
            "session high/low so far. Returns every level with its distance from spot, plus the "
            "nearest overhead resistance (the realistic first target and 'room to run' cap for "
            "a long entry) and the nearest support below (the stop anchor). An empty overhead "
            "list means blue-sky territory -- no structural resistance above."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# --- Advanced level tools (steps 4-6 of the S/R plan): implemented and
# dispatch-wired, but NOT yet exposed to any personality -- except
# analyze_volume_profile_2, which the Volume Signal Detective is built around
# (see VOLUME_DETECTIVE_TOOLS). To enable the others for the Momentum Trader,
# uncomment the three entries in MOMENTUM_TOOLS and the MOMENTUM_SYSTEM_PROMPT
# reassignment under MOMENTUM_ADVANCED_LEVELS_ADDENDUM.

_TOOL_ANALYZE_SWING_LEVELS = {
    "type": "function",
    "function": {
        "name": "analyze_swing_levels",
        "description": (
            "Locate clustered swing-point (fractal) support/resistance in the intraday bars: "
            "confirmed local highs/lows, merged within ~0.25 ATR and ranked by how many times "
            "each level was tested and how recently. A level tested 3+ times is far stronger "
            "than any single extreme print. Returns the ranked clusters plus the nearest swing "
            "resistance above and support below the current price."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "swing": {
                    "type": "integer",
                    "description": (
                        "Bars required on each side to confirm a swing point (default 3, max 10); "
                        "larger means fewer, more significant levels."
                    ),
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_VOLUME_PROFILE = {
    "type": "function",
    "function": {
        "name": "analyze_volume_profile",
        "description": (
            "Build a volume-by-price profile of the intraday bars: the Point of Control (the "
            "price with the most transacted volume -- a magnet/defended level), the 70% value "
            "area, high-volume nodes (support/resistance where positions were built), and "
            "low-volume nodes (air pockets price travels through quickly). An LVN just above "
            "the entry with the next HVN well higher improves the realistic first target."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bins": {
                    "type": "integer",
                    "description": "Number of price slices in the profile (default 24, max 60).",
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Optional trading day to profile, as 'YYYY-MM-DD'. Omit for today's live "
                        "intraday bars. Pass a prior trading date (e.g. the previous session) to "
                        "build the profile from that day's intraday bars instead -- useful for "
                        "comparing today's price against yesterday's POC and value area. Must be an "
                        "actual trading day within roughly the last 60 days; weekends/holidays "
                        "return no data."
                    ),
                },
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_VOLUME_PROFILE_2 = {
    "type": "function",
    "function": {
        "name": "analyze_volume_profile_2",
        "description": (
            "Recent support/resistance from one session's minute-by-minute volume "
            "(yfinance consolidated tape). Finds intraday volume spikes -- local "
            "surges past the opening warm-up -- and classifies each by how price "
            "momentum shifted across it: up-into-flat/down is a SUPPLY line "
            "(distribution/resistance), down-into-flat/up is a DEMAND line "
            "(accumulation/support), a catalyst printing near it flags it news-driven, "
            "else unsure. Then rebuilds a volume-by-price histogram with the spike "
            "volume removed to surface SCATTERED levels (size built up gradually, not "
            "in one burst). Any levels that formed before the most recent news-driven "
            "spike are dropped as stale. Returns the surviving peaks "
            "[{price, time, date, rel_vol, vol, type}] plus nearest support/resistance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bins": {
                    "type": "integer",
                    "description": "Price slices for the scattered-level histogram (default 24, max 60).",
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Optional trading day to analyze, as 'YYYY-MM-DD'. Omit for today's "
                        "live session. A prior date pulls that day's 1-min bars and news from "
                        "history; must be an actual trading day within roughly the last 60 days."
                    ),
                },
            },
            "required": [],
        },
    },
}

_TOOL_GET_FLOOR_PIVOTS = {
    "type": "function",
    "function": {
        "name": "get_floor_pivots",
        "description": (
            "Compute classic floor-trader pivot levels (P, R1-R3, S1-S3) from the prior "
            "completed session's high/low/close. Formula levels rather than structure, but "
            "watched widely enough to act as intraday reaction points; a pivot coinciding with "
            "a structural level (session high, swing cluster, high-volume node) is reinforced. "
            "Returns the levels split around the current price, nearest first."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_OPENING_RANGE = {
    "type": "function",
    "function": {
        "name": "analyze_opening_range",
        "description": (
            "Analyze today's Opening Range Breakout (ORB) setup: the high/low printed in "
            "the 09:30 ET + N minutes window of today's session (measured from bar "
            "timestamps, recovered via a targeted history fetch when the live buffer "
            "doesn't reach back to the open), whether price has since broken out above or "
            "below that range, and whether recent volume confirms the breakout. When the "
            "range genuinely cannot be established the result carries only a `note` -- "
            "there is no valid ORB setup in that case, do not improvise one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {
                    "type": "integer",
                    "description": "Length of the opening range in minutes (default 15).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_GET_SESSION_CLOCK = {
    "type": "function",
    "function": {
        "name": "get_session_clock",
        "description": (
            "Classify the current point in the trading day for breakout timing "
            "discipline: opening window (first 90 min -- historically the most reliable "
            "for breakouts), midday dead zone (12:00-14:00 ET -- notoriously fakeout-"
            "prone), power hour (final hour -- favorable), or outside regular hours. "
            "Returns the ET time, the window label, and `favorable_for_breakouts`. "
            "Check it before arming any breakout entry; in an unfavorable window demand "
            "much stronger confirmation or stand aside."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_PUT_CALL_WALLS = {
    "type": "function",
    "function": {
        "name": "get_put_call_walls",
        "description": (
            "Read the Call Wall (resistance, from peak call open interest) and Put Wall "
            "(support, from peak put open interest) for the ticker, plus the net dealer-gamma "
            "regime (positive = dampening, negative = amplifying) and whether those walls have "
            "been rising/falling recently. Uses the options chain most recently fetched in the "
            "background -- does not fetch fresh data itself. Returns labeled values, actionable "
            "insights, and a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_NEWS = {
    "type": "function",
    "function": {
        "name": "get_news",
        "description": "Get recent news headlines/summaries for the ticker, with impact labels where available.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max number of articles (default 10)."}},
            "required": [],
        },
    },
}

_TOOL_GET_POSITION = {
    "type": "function",
    "function": {
        "name": "get_position",
        "description": (
            "Get the current paper trading position size, cash balance, and total "
            "portfolio value (cash + position marked to the latest price)."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# Bullet list of every watchable field, injected into the alert tool description
# so the model always sees the current, authoritative set.
_ALERT_FIELDS_DOC = "; ".join(f"'{name}' ({desc})" for name, desc in ALERTABLE_FIELDS.items())

_TOOL_SUBMIT_DECISION = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": (
            "Finalize this trading cycle with exactly one decision: buy, sell, or "
            "alert. Must be called exactly once, after analysis is complete. When you "
            "don't want to trade, use 'alert' -- there is no do-nothing action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["buy", "sell", "alert"]},
                "symbol": {
                    "type": "string",
                    "description": (
                        "Ticker to trade -- required for buy/sell (must be one of your "
                        "streamed tickers). Ignored for alert: each alert entry names "
                        "its own symbol."
                    ),
                },
                "quantity": {
                    "type": "number",
                    "description": "Shares to buy/sell. Ignored for alert. Must be > 0 for buy/sell.",
                },
                "regime": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                    "description": "The trading regime established during analysis.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Concise justification covering regime, strategy, and why this action follows from it.",
                },
                "alerts": {
                    "type": "array",
                    "description": (
                        "When action is 'alert': one or more conditions on continuously-updated "
                        "live data that should wake you early -- the instant any one is met -- "
                        "instead of sleeping out the full cycle. Each condition watches one "
                        "field, with 'above' meaning the field reaches or exceeds the value and "
                        "'below' meaning it reaches or falls to the value. Provide several to "
                        "watch a range or multiple signals at once; the first to trigger wakes "
                        f"you. Watchable fields: {_ALERT_FIELDS_DOC}."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Ticker whose live field to watch -- one of your streamed tickers.",
                            },
                            "field": {
                                "type": "string",
                                "enum": list(ALERTABLE_FIELDS.keys()),
                                "description": "Which live state field to watch.",
                            },
                            "condition": {
                                "type": "string",
                                "enum": ["above", "below"],
                                "description": "'above' = field >= value; 'below' = field <= value.",
                            },
                            "value": {
                                "type": "number",
                                "description": "Threshold the field is compared against.",
                            },
                        },
                        "required": ["symbol", "field", "condition", "value"],
                    },
                },
            },
            "required": ["action", "reasoning"],
        },
    },
}

_TACTIC_FIELDS_DOC = "; ".join(f"'{name}' ({desc})" for name, desc in TACTIC_CONDITION_FIELDS.items())

_TOOL_SET_TACTICS = {
    "type": "function",
    "function": {
        "name": "set_tactics",
        "description": (
            "Arm a standing conditional trade plan, executed for you the moment its "
            "conditions are met -- the preferred way to act on concrete levels instead of "
            "trading at the current price or waiting on a bare alert. Each action is a "
            "buy/sell with a size and one or more conditions that must ALL hold "
            "simultaneously; the first action whose conditions are met executes through "
            "the normal paper-fill path, the remaining actions are disarmed, and you are "
            "woken immediately to reevaluate. Replaces any previously armed tactics "
            "(pass an empty actions array to cancel them). Call at most once per cycle, "
            "then still finalize with submit_decision -- with tactics armed, action "
            "'alert' may carry an empty alerts array. On an open long position, a sell "
            "stop (last_price below, armed under your entry price) paired with a sell "
            "take-profit (last_price above) is trailed up automatically as price "
            "advances toward the target; the take-profit level never moves."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": (
                        "Conditional actions, evaluated independently -- e.g. an entry, a "
                        "stop-loss, and a take-profit are three actions. Empty array cancels "
                        "all armed tactics."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["buy", "sell"]},
                            "quantity": {
                                "type": "number",
                                "description": "Shares to trade. Provide exactly one of quantity or quantity_pct.",
                            },
                            "quantity_pct": {
                                "type": "number",
                                "description": (
                                    "Percent (0-100] resolved at execution time: of the current "
                                    "position for a sell, of available cash for a buy. E.g. sell "
                                    "20% of shares, or buy with 50% of cash."
                                ),
                            },
                            "conditions": {
                                "type": "array",
                                "description": (
                                    "Conditions that must ALL hold at the same moment for this "
                                    f"action to execute. Watchable fields: {_TACTIC_FIELDS_DOC}."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {
                                            "type": "string",
                                            "enum": list(TACTIC_CONDITION_FIELDS.keys()),
                                            "description": "Which live field to watch.",
                                        },
                                        "condition": {
                                            "type": "string",
                                            "enum": ["above", "below"],
                                            "description": "'above' = field >= value; 'below' = field <= value.",
                                        },
                                        "value": {
                                            "type": "number",
                                            "description": "Threshold the field is compared against.",
                                        },
                                        "hold_sec": {
                                            "type": "number",
                                            "description": (
                                                "Optional (0-600, default 0): seconds the comparison must hold "
                                                "CONTINUOUSLY before it counts as met. Use on entries to demand a "
                                                "sustained cross instead of firing on a single wick tick (protective "
                                                "stops should usually stay at 0 so they react instantly)."
                                            ),
                                        },
                                    },
                                    "required": ["field", "condition", "value"],
                                },
                            },
                            "note": {
                                "type": "string",
                                "description": "Short label for this leg, e.g. 'entry on retest', 'stop-loss', 'take profit'.",
                            },
                            "trail": {
                                "type": "boolean",
                                "description": (
                                    "Optional (default true): whether the automatic trailing-stop ratchet "
                                    "may raise this action's protective sell stop. Set false to pin a "
                                    "structure stop exactly where you armed it."
                                ),
                            },
                        },
                        "required": ["action", "conditions"],
                    },
                },
                "reasoning": {
                    "type": "string",
                    "description": "Concise justification for the plan: the setup and the levels it encodes.",
                },
            },
            "required": ["actions", "reasoning"],
        },
    },
}

# Only exposed to a strategy agent while it runs UNDER the Automatic orchestrator.
# It lets the strategy relinquish control instead of idling on alerts when the
# regime that suits it has faded -- the orchestrator then re-assesses and may
# activate a better-fitting strategy.
AUTOMATIC_MODE_ADDENDUM = """

--- AUTOMATIC MODE ---
You are running under an Automatic orchestrator that activated you because current \
market conditions favor your strategy. Keep control and trade normally -- exactly \
as described above -- for as long as your edge is plausibly present, including \
standing aside with an alert through ordinary quiet stretches.

But you also have one extra option: stand_down. Call it INSTEAD of submit_decision \
when you judge that the conditions your strategy depends on have genuinely faded \
and your setup is unlikely to appear in the near future -- e.g. a breakout agent in \
a dead, rangebound tape, a mean-reversion agent once a strong trend has taken hold, \
or a momentum agent after the move and its volume have died. Standing down hands \
control back to the orchestrator with your reasoning, so it can re-assess the regime \
and activate a strategy better suited to it.

Judgement: a single slow cycle is NOT a reason to stand down -- that is what a \
normal alert-and-wait is for. Stand down only when the regime itself no longer fits \
your strategy. Standing down does NOT close any open position; if you want to be \
flat before relinquishing, sell first on this cycle and stand down on a later one.
"""

_TOOL_STAND_DOWN = {
    "type": "function",
    "function": {
        "name": "stand_down",
        "description": (
            "Relinquish control back to the Automatic orchestrator because the market "
            "regime no longer fits your strategy and your setup is unlikely to appear "
            "soon. Available only in Automatic mode. Use this INSTEAD of submit_decision "
            "to end the cycle when standing aside on an alert would just be idling in the "
            "wrong regime. Does not close open positions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why your strategy's edge is absent now and unlikely to return soon -- "
                        "cite the regime read (trend/range/volatility/volume) that no longer fits."
                    ),
                },
                "expected_quiet_minutes": {
                    "type": "number",
                    "description": "Rough estimate of how long the drought for your setup is likely to last, in minutes.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

_TOOL_ANALYZE_VWAP_BANDS = {
    "type": "function",
    "function": {
        "name": "analyze_vwap_bands",
        "description": (
            "Analyze today's session VWAP and its volume-weighted standard-deviation bands for "
            "a mean-reversion read: the VWAP, the 1/2/3-sigma bands, price's signed z-score "
            "(how many std devs it sits from VWAP), the ADX trend-strength reading and whether "
            "it confirms a range (below 20), and whether the latest bar is a rejection candle. "
            "The `signal` is 'long_setup' (oversold >= trigger sigma below VWAP in a confirmed "
            "range), 'short_setup' (overbought above VWAP), 'no_setup_trending' (stretched but "
            "ADX shows a trend -- do not fade), or 'no_setup'. Returns labeled values plus a "
            "one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "num_std": {
                    "type": "number",
                    "description": "Std-dev stretch that triggers a setup (default 2.0).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_VWAP_REVERSION_GEOMETRY = {
    "type": "function",
    "function": {
        "name": "vwap_reversion_geometry",
        "description": (
            "Compute the mechanical entry/stop/target math for a VWAP mean-reversion trade. "
            "Target is always VWAP; the stop sits one standard deviation beyond entry (past the "
            "next band). Returns the reward-to-risk ratio and whether it clears the 1.5:1 "
            "mean-reversion minimum. Use this instead of doing the arithmetic yourself before "
            "sizing a trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {"type": "number", "description": "Planned entry price (at/near the band)."},
                "vwap": {"type": "number", "description": "Session VWAP -- the reversion target."},
                "std_dev": {
                    "type": "number",
                    "description": "One standard deviation (the 1σ value from analyze_vwap_bands).",
                },
                "side": {
                    "type": "string",
                    "enum": ["long", "short"],
                    "description": "'long' for a stretch below VWAP, 'short' for a stretch above.",
                },
            },
            "required": ["entry", "vwap", "std_dev"],
        },
    },
}

_TOOL_BREAKOUT_TRADE_GEOMETRY = {
    "type": "function",
    "function": {
        "name": "breakout_trade_geometry",
        "description": (
            "Compute the mechanical entry/stop/target math for a long breakout trade: targets "
            "projected from ATR and/or the base height (1x and 2x each), the resulting "
            "reward-to-risk ratio for each, and whether the best one clears the 2:1 minimum. "
            "Pass the nearest overhead resistance (from get_key_levels) to also get "
            "`room_to_run` -- whether that ceiling sits at least 2x the stop distance above "
            "the entry. Use this instead of doing the arithmetic yourself before sizing a trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {"type": "number", "description": "Planned entry price."},
                "stop": {"type": "number", "description": "Planned stop-loss price, below entry."},
                "atr": {
                    "type": "number",
                    "description": "ATR (from analyze_intraday_momentum), for an ATR-multiple target.",
                },
                "base_height": {
                    "type": "number",
                    "description": (
                        "Height of the consolidation base (from analyze_consolidation), for a "
                        "measured-move target."
                    ),
                },
                "overhead_resistance": {
                    "type": "number",
                    "description": (
                        "Nearest structural resistance level above the entry (from "
                        "get_key_levels), to check whether the trade has room to run before "
                        "hitting a ceiling."
                    ),
                },
            },
            "required": ["entry", "stop"],
        },
    },
}

_TOOL_ANALYZE_ORDER_BLOCKS = {
    "type": "function",
    "function": {
        "name": "analyze_order_blocks",
        "description": (
            "Locate institutional order blocks on the daily (higher) timeframe: bullish demand "
            "zones (the last down candle before an up-move that broke structure) and bearish supply "
            "zones (the mirror). Returns every block with its high/low boundaries, whether it has "
            "been mitigated (already revisited), and how many bars ago it formed, plus the nearest "
            "bullish demand block at/below price (a candidate entry on a return) and the nearest "
            "bearish supply block above (a candidate target). Labeled values plus a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_FAIR_VALUE_GAPS = {
    "type": "function",
    "function": {
        "name": "analyze_fair_value_gaps",
        "description": (
            "Locate fair value gaps (FVGs) -- three-candle price imbalances -- in recent intraday "
            "bars. Returns each gap's boundaries and whether it has been filled, plus the nearest "
            "bullish FVG at/below price (a support imbalance price may be filling now). A held fill "
            "of a bullish FVG is one of the intraday confirmations for a Smart Money long entry. "
            "Labeled values plus a one-line summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent intraday bars to scan (default 50, max 300).",
                }
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_SMART_MONEY_SETUP = {
    "type": "function",
    "function": {
        "name": "analyze_smart_money_setup",
        "description": (
            "The composite Smart Money read: ties a higher-timeframe bullish demand order block "
            "(daily) to today's intraday price action. Returns the demand block being watched, "
            "whether price is inside it, which intraday confirmations are present (bullish "
            "rejection candle, filled bullish FVG, or intraday break-of-structure/breaker), the "
            "suggested entry/stop (just beyond the block)/structural target, the reward-to-risk to "
            "that target, and a `signal`: 'long_setup' (return into demand, confirmed, clears 3:1), "
            "'watching' (valid block but not all conditions met), or 'no_setup'. Plus a `quality` "
            "grade and a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_SMART_MONEY_GEOMETRY = {
    "type": "function",
    "function": {
        "name": "smart_money_trade_geometry",
        "description": (
            "Compute the mechanical entry/stop/target math for a long Smart Money setup: entry at "
            "the demand block on a return, stop just beyond the block, target at the next opposing "
            "structural level. Returns the reward-to-risk ratio and whether it clears the 3:1 "
            "minimum this setup demands (it typically runs 3:1 to 5:1). Use this instead of doing "
            "the arithmetic yourself before sizing a trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {"type": "number", "description": "Planned entry price, inside the order block."},
                "stop": {"type": "number", "description": "Planned stop-loss price, just beyond (below) the block."},
                "target": {"type": "number", "description": "Target price -- the next opposing structural level above entry."},
            },
            "required": ["entry", "stop", "target"],
        },
    },
}

_TOOL_ANALYZE_LIQUIDITY = {
    "type": "function",
    "function": {
        "name": "analyze_liquidity",
        "description": (
            "Map resting liquidity and recent stop-runs on the intraday timeframe -- the core "
            "Smart Money 'stop hunt' read. Returns buy-side liquidity pools above price (clustered "
            "swing highs where buy stops rest) and sell-side pools below (swing lows where sell "
            "stops rest), the nearest of each to price, and whether a recent liquidity SWEEP "
            "occurred: price piercing a prior swing level then closing back through it. A bullish "
            "sweep (a swing low undercut and reclaimed -- a stop-run below support that reversed) "
            "is one of the strongest intraday confirmations for a long off a demand block. Labeled "
            "values plus a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_ANALYZE_PREMIUM_DISCOUNT = {
    "type": "function",
    "function": {
        "name": "analyze_premium_discount",
        "description": (
            "Locate price within the recent daily dealing range: the range high/low, its midpoint "
            "(equilibrium), and whether price sits in the DISCOUNT half (cheap, below equilibrium -- "
            "where Smart Money buys), the PREMIUM half (expensive, above it -- where Smart Money "
            "sells), or at equilibrium. Also returns the deep-discount OTE (optimal trade entry) "
            "zone, the 0.618-0.79 retracement down from the high. A long off a demand block that is "
            "ALSO in discount is higher quality than the same block in premium. Labeled values plus "
            "a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_SMART_MONEY_FLOW = {
    "type": "function",
    "function": {
        "name": "get_smart_money_flow",
        "description": (
            "The institutional 'smart money' ownership footprint for the ticker, from free SEC-"
            "derived disclosures: the percentage of shares held by insiders vs institutions, net "
            "insider buying/selling over the trailing 6 months (Form 4), and the largest "
            "institutional holders with their quarter-over-quarter share changes (13F). This is "
            "slow-moving (quarterly/Form-4 cadence), not an intraday timing signal -- use it as "
            "corroboration: net insider/institutional ACCUMULATION behind a bullish demand block "
            "strengthens the long thesis; DISTRIBUTION is a caution flag. Labeled values plus a "
            "one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_ANALYST_TARGETS = {
    "type": "function",
    "function": {
        "name": "get_analyst_targets",
        "description": (
            "Current Wall Street price targets for the ticker with the actionable read: the "
            "yfinance CONSENSUS (mean/median/high/low target across every covering analyst, the "
            "analyst count, and the recommendation) plus the standing target from UBS, Morgan "
            "Stanley, and Barclays -- each annotated with the implied upside/downside vs the "
            "current price. Use it to gauge how much room the Street sees: price near or above "
            "the consensus mean means limited upside (don't chase a gap into it -- it acts as "
            "resistance/an objective); a wide gap below the mean leaves room to run; price "
            "outside the whole high-low range is a valuation extreme. Targets update at most a "
            "few times a day (cached), so this is positional context, not an intraday trigger. "
            "Returns labeled values, a list of actionable `insights`, and a one-line summary."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_GET_CORPORATE_ACTIONS = {
    "type": "function",
    "function": {
        "name": "get_corporate_actions",
        "description": (
            "Incoming corporate actions scheduled for the ticker over the next two weeks "
            "(configurable): cash/stock dividends with their ex-dividend and payable dates, "
            "forward/reverse splits, mergers, spin-offs, and similar events, flattened into one "
            "chronological list. These are scheduled, mechanical catalysts: an ex-dividend date "
            "lowers the open by roughly the dividend (not a bearish signal), a split resets every "
            "price level, and merger terms can pin or reprice the tape -- check them before "
            "trusting a gap read or leaving tactics armed across an event date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "Lookahead window in days (default 14, max 90).",
                },
            },
            "required": [],
        },
    },
}

_TOOL_ANALYZE_PREMARKET = {
    "type": "function",
    "function": {
        "name": "analyze_premarket",
        "description": (
            "Pre-market read for the upcoming session: the previous close, the latest "
            "pre-market price and the implied opening gap percentage, the pre-market "
            "high/low/volume printed so far from the early bars, and how many minutes "
            "remain until the opening bell. Use it to estimate where the stock will "
            "open before deriving your buy/sell levels."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_SYMBOL_PARAM: dict = {
    "type": "string",
    "description": "Ticker symbol this call applies to -- one of your streamed tickers.",
}


def _add_symbol_param(tool: dict) -> dict:
    """Give a per-ticker tool a required `symbol` argument (in place)."""
    params = tool["function"]["parameters"]
    params["properties"] = {"symbol": copy.deepcopy(_SYMBOL_PARAM), **params.get("properties", {})}
    required = [r for r in (params.get("required") or []) if r != "symbol"]
    params["required"] = ["symbol", *required]
    return tool


# Every tool that reads one ticker's data takes a required `symbol`. The
# exceptions are basket-wide reads (get_position, analyze_market), the pure
# geometry calculators, and the terminal decision tools (which carry symbols
# in their own payloads).
for _tool in (
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_CONSOLIDATION,
    _TOOL_GET_KEY_LEVELS,
    _TOOL_ANALYZE_SWING_LEVELS,
    _TOOL_ANALYZE_VOLUME_PROFILE,
    _TOOL_ANALYZE_VOLUME_PROFILE_2,
    _TOOL_GET_FLOOR_PIVOTS,
    _TOOL_GET_PUT_CALL_WALLS,
    _TOOL_GET_NEWS,
    _TOOL_ANALYZE_VWAP_BANDS,
    _TOOL_ANALYZE_ORDER_BLOCKS,
    _TOOL_ANALYZE_FAIR_VALUE_GAPS,
    _TOOL_ANALYZE_SMART_MONEY_SETUP,
    _TOOL_ANALYZE_LIQUIDITY,
    _TOOL_ANALYZE_PREMIUM_DISCOUNT,
    _TOOL_GET_SMART_MONEY_FLOW,
    _TOOL_GET_ANALYST_TARGETS,
    _TOOL_GET_CORPORATE_ACTIONS,
    _TOOL_ANALYZE_PREMARKET,
    _TOOL_SET_TACTICS,
):
    _add_symbol_param(_tool)


# Momentum trader: RVOL + price action/VWAP + news + price, plus measured levels
# (consolidation base, session-structure S/R) and the R:R geometry check so
# entries anchor to data-based levels -- but still no medium-term regime or
# broad-market backdrop, the whole point is reacting fast to what's happening now.
MOMENTUM_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_CONSOLIDATION,
    _TOOL_GET_KEY_LEVELS,
    _TOOL_BREAKOUT_TRADE_GEOMETRY,
    # Advanced level sources (swing clusters, volume profile, floor pivots).
    # Enabled together with MOMENTUM_ADVANCED_LEVELS_ADDENDUM: step 3 asks for
    # the first level leaving 1.5 ATR of room rather than the nearest tick of
    # session structure, and touch-ranked swing clusters plus volume-profile
    # nodes are what make that level findable.
    _TOOL_ANALYZE_SWING_LEVELS,
    _TOOL_ANALYZE_VOLUME_PROFILE,
    _TOOL_GET_FLOOR_PIVOTS,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# Breakout trader: ORB + session clock discipline + volume + structural levels
# (room-to-run for breakout_trade_geometry) + broad-market backdrop + ATR-based
# targets + news + price.
BREAKOUT_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_GET_SESSION_CLOCK,
    _TOOL_ANALYZE_OPENING_RANGE,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_GET_KEY_LEVELS,
    _TOOL_ANALYZE_MARKET,
    _TOOL_BREAKOUT_TRADE_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# VWAP mean-reversion trader: session VWAP bands + ADX regime gate + volume
# (exhaustion vs breakout) + reversion geometry + news + price. No daily trend
# or options positioning -- this is a fast intraday, regime-gated fade.
REVERSAL_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_VWAP_BANDS,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_VWAP_REVERSION_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# Smart Money (highest-edge): higher-timeframe daily structure (trend + order
# blocks) + intraday confirmation (the composite read + FVG drill-in) + volume +
# SMC geometry + news + price. The composite tool does the heavy lifting; the
# order-block / FVG tools let the agent drill into the structure behind it.
SMART_MONEY_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_ANALYZE_ORDER_BLOCKS,
    _TOOL_ANALYZE_PREMIUM_DISCOUNT,
    _TOOL_ANALYZE_SMART_MONEY_SETUP,
    _TOOL_ANALYZE_FAIR_VALUE_GAPS,
    _TOOL_ANALYZE_LIQUIDITY,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_GET_SMART_MONEY_FLOW,
    _TOOL_GET_ANALYST_TARGETS,
    _TOOL_SMART_MONEY_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# Volume Signal Detective: spike-classified demand/supply lines are the primary
# read (analyze_volume_profile_2), cross-examined against session structure
# (get_key_levels), participation (analyze_volume), the approach into the level
# (analyze_intraday_momentum: ATR + momentum), timing (get_session_clock), and
# the catalyst (get_news) -- with the 2:1 / room-to-run geometry gate before any
# size. No daily trend or options positioning: the edge is one session's tape.
VOLUME_DETECTIVE_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_VOLUME_PROFILE_2,
    _TOOL_ANALYZE_VOLUME,
    _TOOL_ANALYZE_INTRADAY_MOMENTUM,
    _TOOL_GET_KEY_LEVELS,
    _TOOL_GET_SESSION_CLOCK,
    _TOOL_BREAKOUT_TRADE_GEOMETRY,
    _TOOL_GET_NEWS,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

# Premarket analyst: the pre-open read (gap, pre-market range, time to bell) +
# the catalyst + the daily structure and broad backdrop the open will trade
# against. No intraday tools -- there is no session yet; the whole output is a
# set_tactics bracket for the opening prints.
PREMARKET_TOOLS: list[dict] = [
    _TOOL_GET_QUOTE,
    _TOOL_ANALYZE_PREMARKET,
    _TOOL_GET_NEWS,
    _TOOL_GET_CORPORATE_ACTIONS,
    _TOOL_ANALYZE_DAILY_TREND,
    _TOOL_GET_ANALYST_TARGETS,
    _TOOL_ANALYZE_MARKET,
    _TOOL_GET_POSITION,
    _TOOL_SET_TACTICS,
    _TOOL_SUBMIT_DECISION,
]

PERSONALITY_TOOLS: dict[str, list[dict]] = {
    "momentum": MOMENTUM_TOOLS,
    "breakout": BREAKOUT_TOOLS,
    "reversal": REVERSAL_TOOLS,
    "smart_money": SMART_MONEY_TOOLS,
    "volume_detective": VOLUME_DETECTIVE_TOOLS,
    "premarket": PREMARKET_TOOLS,
}


def _log(state: "AppState", entry: dict) -> None:
    entry = {"ts": clock.now().isoformat(), **entry}
    with state.lock:
        state.agent_log.append(entry)


def _quote_age_sec(quote_ts: "str | None") -> "float | None":
    """Seconds elapsed since an RFC-3339 quote timestamp, or None if absent/unparseable."""
    if not quote_ts:
        return None
    try:
        ts = datetime.fromisoformat(str(quote_ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (clock.now() - ts).total_seconds())


def _tool_get_quote(state: "SymbolState") -> dict:
    with state.lock:
        result = {
            "last_price": state.last_price,
            "prev_close": state.prev_close,
            "bid_price": state.bid_price,
            "bid_size": state.bid_size,
            "ask_price": state.ask_price,
            "ask_size": state.ask_size,
            "quote_time": state.quote_ts,
        }

    # The IEX feed's top-of-book is not the consolidated NBBO: off-hours or
    # with an empty IEX book it degrades to a placeholder-wide (or crossed, or
    # hours-old) quote. Surface spread and age, and attach an explicit warning
    # so the agent leans on last_price instead of unexecutable bid/ask levels.
    warnings = []
    bid, ask = result["bid_price"], result["ask_price"]
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2
        result["spread"] = round(ask - bid, 4)
        if mid > 0:
            spread_pct = (ask - bid) / mid * 100
            result["spread_pct"] = round(spread_pct, 3)
            if ask < bid:
                warnings.append("crossed quote (ask below bid) -- bid/ask unreliable, use last_price")
            elif spread_pct > QUOTE_WIDE_SPREAD_PCT:
                warnings.append(
                    f"spread is {spread_pct:.1f}% of the mid -- placeholder-wide quote from a thin "
                    "IEX book (likely pre/post-market); bid/ask are not executable prices, use last_price"
                )
    age = _quote_age_sec(result["quote_time"])
    if age is not None:
        result["quote_age_sec"] = round(age, 1)
        if age > QUOTE_STALE_SEC:
            warnings.append(
                f"quote is {age / 60:.0f} min old (market closed or stream down) -- bid/ask may be stale"
            )
    if warnings:
        result["warning"] = "; ".join(warnings)
    return result


def _bar_dt(bar: dict) -> "datetime | None":
    ts = bar.get("t")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _bar_et_date(bar: dict):
    dt = _bar_dt(bar)
    return dt.astimezone(market_hours.MARKET_TZ).date() if dt else None


def _fetch_prev_session_bars(symbol: str, today_et) -> "list[dict] | None":
    """Yesterday's intraday session for `symbol`, recovered from yfinance when the
    live buffer doesn't reach back that far. Walks back up to a week so weekends
    and holidays resolve to the last actual trading day."""
    for back in range(1, 6):
        day = today_et - timedelta(days=back)
        try:
            prev = historical.fetch_intraday_bars_for_date(symbol, day.isoformat())
        except Exception:
            prev = []
        if prev:
            return prev
    return None


def _market_window_return(bars: list[dict]) -> "float | None":
    """SPY's % move over the same clock window as `bars`, for beta-adjustment.

    Aligns to the recent window by timestamp so the market move being removed is
    contemporaneous with the ticker's; falls back to the same bar count when
    timestamps are missing (e.g. synthetic bars)."""
    try:
        spy = historical.fetch_intraday_volume_bars(historical.SPY_SYMBOL)
    except Exception:
        spy = []
    if not spy or len(spy) < 2:
        return None
    first_t, last_t = _bar_dt(bars[0]), _bar_dt(bars[-1])
    window = spy
    if first_t and last_t:
        window = [b for b in spy if (bt := _bar_dt(b)) and first_t <= bt <= last_t]
    if len(window) < 2:
        window = spy[-len(bars):]
    if len(window) < 2:
        return None
    start = float(window[0]["c"])
    end = float(window[-1]["c"])
    return (end / start - 1) * 100 if start else None


def _tool_analyze_intraday_momentum(state: "SymbolState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 50), 300))
    with state.lock:
        all_bars = list(state.bars)
    if not all_bars:
        return {"note": "no intraday bars available yet"}
    bars = all_bars[-n:]

    # Split the live buffer into today's and yesterday's ET sessions so the
    # analysis can report today's total move and yesterday's momentum.
    today_et = _bar_et_date(all_bars[-1])
    full_session_bars = None
    prev_session_bars = None
    if today_et is not None:
        full_session_bars = [b for b in all_bars if _bar_et_date(b) == today_et]
        prior_dates = sorted(
            {d for b in all_bars if (d := _bar_et_date(b)) is not None and d < today_et}
        )
        if prior_dates:
            prev_et = prior_dates[-1]
            prev_session_bars = [b for b in all_bars if _bar_et_date(b) == prev_et]
        if not prev_session_bars:
            prev_session_bars = _fetch_prev_session_bars(state.symbol, today_et)

    # Market-neutral inputs: the ticker's long-term beta to SPY and SPY's move
    # over the same recent window. Best-effort -- any fetch failure just omits
    # the market-neutral block rather than failing the whole read.
    market_return_pct = None
    beta_val = None
    beta_window = None
    try:
        beta_info = historical.fetch_market_beta(state.symbol)
        if beta_info:
            market_return_pct = _market_window_return(bars)
            if market_return_pct is not None:
                beta_val = beta_info["beta"]
                beta_window = beta_info["window_days"]
    except Exception:
        market_return_pct = None
        beta_val = None

    return ta.analyze_intraday(
        bars,
        prev_session_bars=prev_session_bars,
        full_session_bars=full_session_bars,
        market_return_pct=market_return_pct,
        beta=beta_val,
        beta_window_days=beta_window,
    )


def _tool_analyze_daily_trend(state: "SymbolState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 60), 365))
    bars = list(state.daily_bars)[-n:]
    if not bars:
        return {"note": "no daily bars available yet"}
    return ta.analyze_trend(bars)


def _opening_range_for(state: "SymbolState", minutes: int, allow_fetch: bool = True) -> "dict | None":
    """Today's opening range for one symbol, from the most reliable source
    available: the per-symbol cache, else measured from the live bar buffer,
    else recovered with a targeted REST fetch of the 09:30 ET window (which
    survives buffer eviction and mid-session starts). Completed ranges are
    cached on the SymbolState. Returns None when the range genuinely cannot
    be established -- never a fabricated range."""
    now = clock.now()
    today_et = now.astimezone(market_hours.MARKET_TZ).date()

    cached = state.opening_range
    if (
        cached
        and cached.get("date") == today_et.isoformat()
        and cached.get("minutes") == minutes
        and cached.get("complete")
    ):
        return cached

    with state.lock:
        bars = list(state.bars)
    rng = ta.compute_opening_range(bars, minutes)
    if "high" not in rng and allow_fetch and state.api_key and state.api_secret:
        open_utc = datetime.combine(
            today_et, market_hours.MARKET_OPEN, tzinfo=market_hours.MARKET_TZ
        ).astimezone(timezone.utc)
        if now >= open_utc:
            try:
                window = fetch_bars_window(
                    state.symbol,
                    "1Min",
                    open_utc,
                    open_utc + timedelta(minutes=minutes),
                    state.api_key,
                    state.api_secret,
                    state.feed,
                )
            except Exception:
                window = []
            if window:
                rng = ta.compute_opening_range(window, minutes, assume_coverage=True)
    if "high" not in rng:
        return None
    if rng.get("complete"):
        state.opening_range = rng
    return rng


def breakout_preconditions(app: "AppState", symbols: list[str], minutes: int = 15) -> "str | None":
    """Deterministic gate for activating the Breakout strategy: returns the
    reason it is NOT currently tradeable, or None when it is.

    The Breakout Trader's whole edge hangs on a real, measurable opening range
    and a session window where breaks tend to follow through -- activating it
    in the midday dead zone, or when no symbol's 09:30 ET window can be
    established, deploys it into a tape it was never designed for.
    """
    clock = ta.session_time_window()
    if not clock.get("favorable_for_breakouts"):
        return f"breakout is not selectable right now: {clock.get('summary')}"
    for symbol in symbols:
        ss = app.sym(symbol)
        if ss is not None and _opening_range_for(ss, minutes) is not None:
            return None
    return (
        "breakout is not selectable: no symbol has a measurable opening range for "
        "today's session (bar history does not reach back to the 09:30 ET open and "
        "the opening window could not be recovered)"
    )


def _tool_analyze_opening_range(state: "SymbolState", minutes: object = None) -> dict:
    n = max(1, min(int(minutes or 15), 120))
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    rng = _opening_range_for(state, n)
    if rng is None:
        # Fall through with the honest measurement note (no cache, no fetch).
        return ta.analyze_opening_range(bars, minutes=n)
    return ta.analyze_opening_range(bars, minutes=n, opening_range=rng)


def _tool_analyze_market(state: "AppState") -> dict:
    data = historical.fetch_market_indicators()
    return ta.analyze_market(
        vix_close=data.get("vix"),
        spy_close=data.get("spy"),
        vix3m_close=data.get("vix3m"),
    )


def _tool_analyze_volume(state: "SymbolState") -> dict:
    # Volume is read from yfinance (consolidated tape across every exchange)
    # rather than Alpaca's single-venue IEX feed, whose few-percent tape share
    # makes absolute volume and volume ratios unreliable. Today's cumulative
    # volume and the ADV baseline both come from yfinance so rvol_pace stays a
    # like-for-like ratio. Fall back to Alpaca's own bars only when yfinance is
    # unavailable.
    yf_bars = historical.fetch_intraday_volume_bars(state.symbol)
    if yf_bars:
        day_volume = sum(float(b.get("v") or 0.0) for b in yf_bars)
        daily_bars = historical.fetch_daily_volume_bars(state.symbol) or state.daily_bars
        pace = rvol_pace(day_volume, daily_bars)
        return ta.analyze_volume(yf_bars, rvol_pace=pace, partial_volume_feed=False)
    with state.lock:
        bars = list(state.bars)
        day_volume = state.day_volume
    if not bars:
        return {"note": "no intraday bars available yet"}
    pace = rvol_pace(day_volume, state.daily_bars)
    return ta.analyze_volume(bars, rvol_pace=pace, partial_volume_feed=state.feed == "iex")


def _tool_analyze_consolidation(state: "SymbolState", base_bars: object = None) -> dict:
    n = max(5, min(int(base_bars or 10), 60))
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_consolidation(bars, base_bars=n)


def _tool_get_key_levels(state: "SymbolState") -> dict:
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
    daily = list(state.daily_bars)
    # Same cached/recovered range the ORB tool uses, so the opening-range
    # levels here can never disagree with analyze_opening_range's.
    rng = _opening_range_for(state, 15)
    return ta.key_levels(bars, daily_bars=daily, spot=spot, opening_range=rng)


def _tool_analyze_swing_levels(state: "SymbolState", swing: object = None) -> dict:
    k = max(2, min(int(swing or 3), 10))
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.swing_levels(bars, swing=k, spot=spot)


def _tool_analyze_volume_profile(state: "SymbolState", bins: object = None, date: object = None) -> dict:
    n = max(8, min(int(bins or 24), 60))
    day = str(date or "").strip()
    if day:
        # A prior session: pull that day's intraday bars from yfinance rather than
        # today's live buffer. spot stays the live price so the profile still
        # reports where the current price sits relative to the old day's levels.
        try:
            bars = historical.fetch_intraday_bars_for_date(state.symbol, day)
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception:
            return {"note": f"could not fetch intraday bars for {day}"}
        if not bars:
            return {"note": f"no intraday bars for {day} (non-trading day, or outside yfinance's ~60-day window)"}
        with state.lock:
            spot = state.last_price
        result = ta.volume_profile_levels(bars, bins=n, spot=spot)
        result["date"] = day
        return result
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.volume_profile_levels(bars, bins=n, spot=spot)


def _news_times_for_date(state: "SymbolState", session_date) -> list[str]:
    """ISO timestamps of `symbol`'s news for one ET session, for the spike
    news-driven classification. Today's session reads the already-loaded live
    news; a prior session is recovered with a dated Alpaca news query. Best
    effort -- any failure just yields no timestamps (spikes fall back to
    supply/demand/unsure)."""
    today_et = clock.now().astimezone(market_hours.MARKET_TZ).date()
    if session_date >= today_et:
        with state.lock:
            return [str(item.get("created_at")) for item in state.news if item.get("created_at")]
    if not (state.api_key and state.api_secret):
        return []
    start = datetime.combine(session_date, dt_time(0, 0), tzinfo=market_hours.MARKET_TZ)
    end = start + timedelta(days=1)
    try:
        articles = fetch_news_window(state.symbol, start, end, state.api_key, state.api_secret)
    except Exception:
        return []
    return [str(item.get("created_at")) for item in articles if item.get("created_at")]


def _tool_analyze_volume_profile_2(state: "SymbolState", bins: object = None, date: object = None) -> dict:
    n = max(8, min(int(bins or 24), 60))
    day = str(date or "").strip()
    with state.lock:
        spot = state.last_price
    if day:
        # A prior session: minute bars for that ET day from yfinance (same
        # consolidated-tape source as today's live volume).
        try:
            bars = historical.fetch_intraday_bars_for_date(state.symbol, day)
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception:
            return {"note": f"could not fetch intraday bars for {day}"}
        if not bars:
            return {"note": f"no intraday bars for {day} (non-trading day, or outside yfinance's ~60-day window)"}
        try:
            session_date = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError as exc:
            return {"error": str(exc)}
    else:
        # Today's live session: consolidated-tape 1-min bars from yfinance so
        # the volume series is accurate, not Alpaca's single-venue IEX feed.
        bars = historical.fetch_intraday_volume_bars(state.symbol)
        if not bars:
            with state.lock:
                bars = list(state.bars)
        if not bars:
            return {"note": "no intraday bars available yet"}
        session_date = clock.now().astimezone(market_hours.MARKET_TZ).date()

    news_times = _news_times_for_date(state, session_date)
    return ta.analyze_volume_profile_2(
        bars,
        news_times=news_times,
        date=day or None,
        spot=spot,
        price_bins=n,
    )


def _tool_get_floor_pivots(state: "SymbolState") -> dict:
    with state.lock:
        spot = state.last_price
    daily = list(state.daily_bars)
    return ta.floor_pivots(daily, spot=spot)


def _tool_get_put_call_walls(state: "SymbolState") -> dict:
    with state.lock:
        data = state.options_chain
        history = list(state.options_wall_history)
    if not data:
        return {"note": "no options chain data available yet"}
    return ta.get_put_call_walls_and_gamma(
        strikes=data["strikes"],
        calls_oi=data["calls_oi"],
        puts_oi=data["puts_oi"],
        calls_gamma_exposure=data["calls_gamma_exposure"],
        puts_gamma_exposure=data["puts_gamma_exposure"],
        spot=data["spot"],
        wall_history=history,
    )


def _tool_get_news(state: "SymbolState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 10), 30))
    with state.lock:
        news = list(state.news)[:n]
        impacts = dict(state.news_impacts)
    return {
        "articles": [
            {
                "headline": item.get("headline"),
                "summary": item.get("summary"),
                "created_at": item.get("created_at"),
                "source": item.get("source"),
                "impact": impacts.get(str(item.get("id", "")), "unknown"),
            }
            for item in news
        ]
    }


def _tool_get_position(app: "AppState", tracker: "DecisionTracker") -> dict:
    snap = tracker.snapshot()
    # Standing conditional orders still armed from a previous cycle, per ticker;
    # a set_tactics call replaces that ticker's plan, actions=[] cancels it.
    armed = {
        ss.symbol: tactics_summaries(ss.tactics)
        for ss in app.iter_symbol_states()
        if ss.tactics is not None
    }
    return {
        "cash": snap["cash"],
        "positions": {sym: qty for sym, qty in snap["positions"].items() if qty},
        # Kept fresh independently by the price stream, not fetched here.
        "portfolio_value": app.portfolio_value,
        "decisions_so_far": len(snap["decisions"]),
        "armed_tactics": armed or None,
    }


def _resolve_symbol_state(app: "AppState", args: dict) -> "tuple[SymbolState | None, str | None]":
    """Resolve a tool call's `symbol` argument to its SymbolState. A missing
    symbol falls back to the sole streamed ticker; otherwise it must name one
    of the streamed tickers. Returns (state, None) or (None, error)."""
    raw = str(args.get("symbol") or "").strip().upper()
    if not raw and len(app.symbols) == 1:
        raw = app.symbols[0]
    state = app.sym(raw) if raw else None
    if state is None:
        return None, (
            f"unknown or missing symbol {raw!r}; pass one of your streamed tickers: "
            f"{', '.join(app.symbols) or '(none)'}"
        )
    return state, None


def _handle_set_tactics(args: dict, app: "AppState", tracker: "DecisionTracker") -> dict:
    """Arm (or cancel) one symbol's tactics as requested by a set_tactics tool call."""
    state, error = _resolve_symbol_state(app, args)
    if error is not None:
        return {"error": error}
    symbol = state.symbol
    raw_actions = args.get("actions")
    reasoning = str(args.get("reasoning") or "")

    if isinstance(raw_actions, list) and not raw_actions:
        had = tactics_summaries(state.tactics)
        state.tactics = None
        _log(app, {"type": "tactics_set", "symbol": symbol, "cancelled": had, "reasoning": reasoning})
        return {"status": "cancelled", "symbol": symbol, "cancelled_tactics": had}

    tactics, error = normalize_tactics(symbol, raw_actions, reasoning)
    if error is not None:
        return {"error": error}

    replaced = tactics_summaries(state.tactics)
    state.tactics = tactics
    summaries = tactics_summaries(tactics)
    with state.lock:
        price = state.last_price
    # Recorded as a no-op "tactics" decision so the arming moment shows up on
    # the portfolio-value chart and in the decision history/report.
    tracker.record_tactics(symbol, summaries, reasoning, price)
    _log(app, {"type": "tactics_set", "symbol": symbol, "tactics": summaries, "replaced": replaced, "reasoning": reasoning})
    return {
        "status": "armed",
        "symbol": symbol,
        "tactics": summaries,
        "replaced_tactics": replaced or None,
        "note": (
            "You are woken the instant any action executes (that ticker's remaining "
            "actions are disarmed). Now finalize the cycle with submit_decision -- "
            "action 'alert' may carry an empty alerts array while tactics are armed."
        ),
    }


def _tool_breakout_trade_geometry(
    entry: object,
    stop: object,
    atr: object = None,
    base_height: object = None,
    overhead_resistance: object = None,
) -> dict:
    return ta.breakout_trade_geometry(
        float(entry),
        float(stop),
        base_height=float(base_height) if base_height is not None else None,
        atr=float(atr) if atr is not None else None,
        overhead_resistance=float(overhead_resistance) if overhead_resistance is not None else None,
    )


def _tool_analyze_vwap_bands(state: "SymbolState", num_std: object = None) -> dict:
    with state.lock:
        bars = list(state.bars)
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_vwap_bands(bars, num_std=float(num_std) if num_std is not None else 2.0)


def _tool_vwap_reversion_geometry(entry: object, vwap: object, std_dev: object, side: object = None
) -> dict:
    return ta.vwap_reversion_geometry(
        float(entry),
        float(vwap),
        float(std_dev),
        side=str(side) if side is not None else "long",
    )


def _tool_analyze_order_blocks(state: "SymbolState") -> dict:
    bars = list(state.daily_bars)
    if not bars:
        return {"note": "no daily bars available yet"}
    with state.lock:
        spot = state.last_price
    return ta.analyze_order_blocks(bars, spot=spot)


def _tool_analyze_fair_value_gaps(state: "SymbolState", limit: object = None) -> dict:
    n = max(1, min(int(limit or 50), 300))
    with state.lock:
        bars = list(state.bars)[-n:]
        spot = state.last_price
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_fair_value_gaps(bars, spot=spot)


def _tool_analyze_smart_money_setup(state: "SymbolState") -> dict:
    daily = list(state.daily_bars)
    if not daily:
        return {"note": "no daily bars available yet"}
    with state.lock:
        intraday = list(state.bars)
        spot = state.last_price
    return ta.analyze_smart_money_setup(daily, intraday_bars=intraday, spot=spot)


def _tool_smart_money_trade_geometry(entry: object, stop: object, target: object) -> dict:
    return ta.smart_money_trade_geometry(float(entry), float(stop), float(target))


def _tool_analyze_liquidity(state: "SymbolState") -> dict:
    with state.lock:
        bars = list(state.bars)
        spot = state.last_price
    if not bars:
        return {"note": "no intraday bars available yet"}
    return ta.analyze_liquidity(bars, spot=spot)


def _tool_analyze_premium_discount(state: "SymbolState") -> dict:
    daily = list(state.daily_bars)
    if not daily:
        return {"note": "no daily bars available yet"}
    with state.lock:
        spot = state.last_price
    return ta.analyze_premium_discount(daily, spot=spot)


def _tool_get_smart_money_flow(state: "SymbolState") -> dict:
    symbol = state.symbol
    if not symbol:
        return {"note": "no symbol set"}
    return historical.fetch_smart_money_flow(symbol)


def _tool_get_analyst_targets(state: "SymbolState") -> dict:
    symbol = state.symbol
    if not symbol:
        return {"note": "no symbol set"}
    # Pass the live streamed price so the upside math is anchored to the current
    # tape rather than Yahoo's slower quote.
    with state.lock:
        spot = state.last_price
    return historical.fetch_analyst_targets(symbol, current_price=spot)


def _tool_get_corporate_actions(state: "SymbolState", days_ahead: object = None) -> dict:
    symbol = state.symbol
    if not symbol:
        return {"note": "no symbol set"}
    key, secret = state.api_key, state.api_secret
    if not key or not secret:
        return {"note": "Alpaca API keys are not configured; corporate actions unavailable"}
    days = max(1, min(int(days_ahead or 14), 90))
    actions = fetch_corporate_actions(symbol, key, secret, days_ahead=days)
    if not actions:
        return {"note": f"no corporate actions scheduled for {symbol} in the next {days} days"}
    return {"window_days": days, "upcoming_corporate_actions": actions}


def _tool_analyze_premarket(state: "SymbolState") -> dict:
    now = clock.now()
    # The session the read is about: the one in progress (edge case: the bell
    # already rang while the analyst was reasoning) or the upcoming one.
    open_dt = market_hours.session_open(now) or market_hours.next_market_open(now)
    with state.lock:
        bars = list(state.bars)
        last_price = state.last_price
        prev_close = state.prev_close

    result: dict = {
        "market_is_open": market_hours.is_market_open(now),
        "market_open_at": open_dt.isoformat(),
        "minutes_until_open": round(max(0.0, (open_dt - now).total_seconds()) / 60.0, 1),
        "prev_close": prev_close,
        "last_price": last_price,
    }
    if last_price is not None and prev_close:
        result["implied_gap_pct"] = round((last_price / prev_close - 1.0) * 100.0, 2)

    # Pre-market bars: same trading day as the open, printed before the bell.
    session_date = open_dt.astimezone(market_hours.MARKET_TZ).date()
    pre_bars = []
    for bar in bars:
        try:
            ts = datetime.fromisoformat(str(bar["t"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        if ts < open_dt and ts.astimezone(market_hours.MARKET_TZ).date() == session_date:
            pre_bars.append(bar)
    try:
        if pre_bars:
            result["premarket_session"] = {
                "bars": len(pre_bars),
                "high": max(float(b["h"]) for b in pre_bars),
                "low": min(float(b["l"]) for b in pre_bars),
                "volume": sum(float(b.get("v") or 0.0) for b in pre_bars),
                "last_bar_close": float(pre_bars[-1]["c"]),
                "last_bar_time": pre_bars[-1].get("t"),
            }
        else:
            result["premarket_session"] = {
                "note": "no pre-market bars for the upcoming session yet"
            }
    except (KeyError, TypeError, ValueError):
        result["premarket_session"] = {"note": "pre-market bars are malformed"}
    return result


def _per_symbol(handler: "Callable[..., dict]", *arg_names: str) -> "Callable[[dict, AppState, DecisionTracker], dict]":
    """Wrap a SymbolState-reading tool helper: resolve the call's `symbol` to
    its SymbolState (erroring on unknown tickers), then forward the named args."""

    def run(args: dict, app: "AppState", tracker: "DecisionTracker") -> dict:
        state, error = _resolve_symbol_state(app, args)
        if error is not None:
            return {"error": error}
        return handler(state, *[args.get(name) for name in arg_names])

    return run


_DISPATCH: dict[str, Callable[[dict, "AppState", "DecisionTracker"], dict]] = {
    "get_quote": _per_symbol(_tool_get_quote),
    "analyze_intraday_momentum": _per_symbol(_tool_analyze_intraday_momentum, "limit"),
    "analyze_daily_trend": _per_symbol(_tool_analyze_daily_trend, "limit"),
    "analyze_opening_range": _per_symbol(_tool_analyze_opening_range, "minutes"),
    "analyze_market": lambda args, app, tracker: _tool_analyze_market(app),
    "analyze_volume": _per_symbol(_tool_analyze_volume),
    "analyze_consolidation": _per_symbol(_tool_analyze_consolidation, "base_bars"),
    "get_key_levels": _per_symbol(_tool_get_key_levels),
    "analyze_swing_levels": _per_symbol(_tool_analyze_swing_levels, "swing"),
    "analyze_volume_profile": _per_symbol(_tool_analyze_volume_profile, "bins", "date"),
    "analyze_volume_profile_2": _per_symbol(_tool_analyze_volume_profile_2, "bins", "date"),
    "get_floor_pivots": _per_symbol(_tool_get_floor_pivots),
    "breakout_trade_geometry": lambda args, app, tracker: _tool_breakout_trade_geometry(
        args.get("entry"),
        args.get("stop"),
        args.get("atr"),
        args.get("base_height"),
        args.get("overhead_resistance"),
    ),
    "analyze_vwap_bands": _per_symbol(_tool_analyze_vwap_bands, "num_std"),
    "vwap_reversion_geometry": lambda args, app, tracker: _tool_vwap_reversion_geometry(
        args.get("entry"), args.get("vwap"), args.get("std_dev"), args.get("side")
    ),
    "analyze_order_blocks": _per_symbol(_tool_analyze_order_blocks),
    "analyze_fair_value_gaps": _per_symbol(_tool_analyze_fair_value_gaps, "limit"),
    "analyze_smart_money_setup": _per_symbol(_tool_analyze_smart_money_setup),
    "analyze_liquidity": _per_symbol(_tool_analyze_liquidity),
    "analyze_premium_discount": _per_symbol(_tool_analyze_premium_discount),
    "get_smart_money_flow": _per_symbol(_tool_get_smart_money_flow),
    "get_analyst_targets": _per_symbol(_tool_get_analyst_targets),
    "get_corporate_actions": _per_symbol(_tool_get_corporate_actions, "days_ahead"),
    "analyze_premarket": _per_symbol(_tool_analyze_premarket),
    "smart_money_trade_geometry": lambda args, app, tracker: _tool_smart_money_trade_geometry(
        args.get("entry"), args.get("stop"), args.get("target")
    ),
    "get_put_call_walls": _per_symbol(_tool_get_put_call_walls),
    "get_session_clock": lambda args, app, tracker: ta.session_time_window(),
    "get_news": _per_symbol(_tool_get_news, "limit"),
    "get_position": lambda args, app, tracker: _tool_get_position(app, tracker),
}


def _dispatch_tool(name: str, args: dict, app: "AppState", tracker: "DecisionTracker") -> dict:
    handler = _DISPATCH.get(name)
    if handler is None:
        result = {"error": f"unknown tool {name}"}
    else:
        try:
            result = handler(args, app, tracker)
        except Exception as exc:
            result = {"error": str(exc)}
    scoring.record_tool_call(app, name, result)
    return result


def _reject(messages: list[dict], tool_call_id: str, error: str) -> None:
    """Hand a malformed submit_decision back to the model as a tool error so it can
    retry. Used for the cases that have no valid resting state -- an empty alert, a
    zero-quantity trade, or an unrecognized action -- since there is no longer a
    do-nothing decision to silently fall back to."""
    messages.append(
        {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps({"error": error})}
    )


@obs.observe(name="agent-cycle")
def run_agent_cycle(
    client: Any,
    model: str,
    symbols: list[str],
    state: "AppState",
    tracker: "DecisionTracker",
    max_iters: int = AGENT_MAX_TOOL_ITERS,
    personality: str = DEFAULT_PERSONALITY,
    under_automatic: bool = False,
    system_prompt_override: "str | None" = None,
) -> str:
    """Run one analyze-then-decide cycle over the whole symbol basket. Always
    ends with exactly one recorded decision.

    When Langfuse is configured, the whole cycle is one trace: every LLM turn
    nests under it as a generation, so per-cycle latency, token usage, and cost
    roll up automatically (see `agent_stonks.observability`).

    When `under_automatic` is True the strategy is running under the Automatic
    orchestrator: it also gets a `stand_down` tool to relinquish control when the
    regime no longer fits it. Returns "stand_down" in that case (so the
    orchestrator can re-assess and pick another strategy), or "decided" when the
    cycle finalized with a normal buy/sell/alert (or the forced-sleep fallback).
    """
    symbols_label = ", ".join(symbols)
    obs.update_trace(
        name=f"agent-cycle:{symbols_label}",
        input=symbols_label,
        metadata={"model": model, "symbols": symbols_label, "personality": personality},
    )
    # An override replaces only the personality's base prompt (simlab's editable
    # prompts); the operational addenda below are always appended unchanged.
    system_prompt = system_prompt_override or AGENT_PERSONALITIES.get(
        personality, AGENT_PERSONALITIES[DEFAULT_PERSONALITY]
    )["system_prompt"]
    system_prompt = (
        system_prompt
        + MULTI_SYMBOL_ADDENDUM.format(symbols=symbols_label)
        + TACTICS_ADDENDUM
    )
    if personality != PREMARKET_PERSONALITY:
        system_prompt = system_prompt + _session_closed_addendum()
    tools = PERSONALITY_TOOLS.get(personality, MOMENTUM_TOOLS)
    if under_automatic:
        system_prompt = system_prompt + AUTOMATIC_MODE_ADDENDUM
        tools = [*tools, _TOOL_STAND_DOWN]
    state.clear_alerts()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Tickers: {symbols_label}. Run your analysis process and finish by calling submit_decision.",
        },
    ]
    _log(state, {"type": "cycle_start", "text": f"Starting analysis cycle for {symbols_label}"})

    decision_made = False
    stood_down = False
    for _ in range(max_iters):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto"
            )
        except Exception as exc:
            _log(state, {"type": "error", "text": f"LLM call failed: {exc}"})
            break

        msg = response.choices[0].message
        if msg.content:
            _log(state, {"type": "analysis", "text": msg.content})

        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            calls = []
            for tc in tool_calls:
                call = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                # Gemini 3+ "thinking" models attach a thought_signature here that must be
                # echoed back verbatim on the next turn, or the API rejects the request with
                # "Function call ... is missing a thought_signature".
                extra_content = getattr(tc, "extra_content", None)
                if extra_content:
                    call["extra_content"] = extra_content
                calls.append(call)
            assistant_msg["tool_calls"] = calls
        messages.append(assistant_msg)

        if not tool_calls:
            messages.append(
                {"role": "user", "content": "Please finalize this cycle by calling submit_decision now."}
            )
            continue

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "stand_down" and under_automatic:
                reasoning = args.get("reasoning", "")
                quiet = args.get("expected_quiet_minutes")
                _log(
                    state,
                    {
                        "type": "stand_down",
                        "personality": personality,
                        "reasoning": reasoning,
                        "expected_quiet_minutes": quiet,
                    },
                )
                obs.update_trace(output={"action": "stand_down", "reasoning": reasoning})
                # The relinquishing strategy's conditional orders must not keep
                # trading under whatever regime/strategy comes next.
                for ss in state.iter_symbol_states():
                    ss.tactics = None
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"status": "relinquished"})}
                )
                stood_down = True
                decision_made = True
                break

            if name == "set_tactics":
                # _handle_set_tactics writes its own "tactics_set" log entry on
                # success; only a validation failure is logged as a plain tool call.
                result = _handle_set_tactics(args, state, tracker)
                scoring.record_tactics_call(state, ok="error" not in result)
                result_content = json.dumps(result)
                if "error" in result:
                    _log(state, {"type": "tool_call", "name": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})
                continue

            if name == "submit_decision":
                action = args.get("action", "")
                quantity = float(args.get("quantity") or 0)
                reasoning = args.get("reasoning", "")
                regime = args.get("regime", "unknown")
                default_symbol = symbols[0] if len(symbols) == 1 else None
                # Validate each requested condition against the watchable-field
                # and streamed-symbol registries; silently drop malformed specs
                # so one bad entry doesn't sink a valid bracket.
                alerts = [
                    a
                    for a in (
                        normalize_alert(r, symbols=symbols, default_symbol=default_symbol)
                        for r in (args.get("alerts") or [])
                    )
                    if a is not None
                ]

                if action == "alert" and not alerts and not state.any_tactics():
                    # Some models (small/cheap ones especially) pick action="alert" but
                    # forget the conditions, or name a field that isn't watchable. Reject
                    # and let the model retry instead of silently recording a no-op -- it
                    # never sees that happen otherwise, so it can't course-correct.
                    _reject(
                        messages,
                        tc.id,
                        "action 'alert' requires a non-empty 'alerts' array, each "
                        f"entry having symbol (one of: {', '.join(symbols)}), field "
                        f"(one of: {', '.join(ALERTABLE_FIELDS)}), condition ('above' or "
                        "'below'), and a numeric value. Call submit_decision again "
                        "with at least one valid condition (or arm a plan with "
                        "set_tactics first -- an empty alerts array is only allowed "
                        "while tactics are armed). Standing aside always means "
                        "setting an alert -- there is no do-nothing action.",
                    )
                    continue

                if action in ("buy", "sell") and quantity <= 0:
                    # A trade with no size is not a trade. Don't silently record a
                    # no-op -- make the model either commit to a size or stand aside
                    # explicitly with an alert.
                    _reject(
                        messages,
                        tc.id,
                        f"action '{action}' requires a quantity greater than 0. Call "
                        "submit_decision again with a positive quantity, or use action "
                        "'alert' with one or more conditions to watch if you don't want "
                        "to trade right now.",
                    )
                    continue

                if action in ("buy", "sell"):
                    trade_state, symbol_error = _resolve_symbol_state(app=state, args=args)
                    if symbol_error is not None:
                        _reject(
                            messages,
                            tc.id,
                            f"action '{action}' requires a valid 'symbol': {symbol_error}. "
                            "Call submit_decision again with the ticker to trade.",
                        )
                        continue
                    decision = tracker.record_trade(
                        trade_state.symbol, action, quantity, reasoning,
                        state.api_key, state.api_secret, state.feed,
                    )
                elif action == "alert":
                    involved = sorted({a["symbol"] for a in alerts}) or list(symbols)
                    decision = tracker.record_alert(", ".join(involved), alerts, reasoning)
                    # Distribute each condition to the SymbolState whose stream
                    # watches it (cycle start cleared the previous set).
                    for a in alerts:
                        alert_state = state.sym(a["symbol"])
                        if alert_state is not None:
                            alert_state.alerts = [*alert_state.alerts, a]
                else:
                    # Unknown / removed action (e.g. a model still reaching for the old
                    # "sleep"). There is no do-nothing path: reject and retry.
                    _reject(
                        messages,
                        tc.id,
                        "action must be one of 'buy', 'sell', or 'alert'. To stand aside "
                        "without trading, use action 'alert' with one or more conditions "
                        "to watch -- there is no 'sleep' or do-nothing action. Call "
                        "submit_decision again.",
                    )
                    continue
                _log(
                    state,
                    {
                        "type": "decision",
                        "action": decision.action,
                        "symbol": decision.symbol,
                        "status": decision.status,
                        "price": decision.price,
                        "quantity": decision.filled_quantity,
                        "reasoning": reasoning,
                        "regime": regime,
                        "alerts": decision.alerts,
                    },
                )
                result_content = json.dumps(
                    {
                        "status": decision.status,
                        "filled_quantity": decision.filled_quantity,
                        "price": decision.price,
                        "cash_after": decision.cash_after,
                        "position_after": decision.position_after,
                    }
                )
                decision_made = True
                obs.update_trace(
                    output={"action": decision.action, "regime": regime, "reasoning": reasoning}
                )
            else:
                result = _dispatch_tool(name, args, state, tracker)
                result_content = json.dumps(result)
                _log(state, {"type": "tool_call", "name": name, "args": args, "result": result})

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})

        if decision_made:
            break

    # Deterministic numeric-faithfulness check over the cycle's full transcript
    # (see agent_stonks.scoring); aggregated into the daily scoring session.
    scoring.record_cycle_grounding(state, messages, personality)

    if stood_down:
        return "stand_down"

    if not decision_made:
        forced = tracker.record_sleep(
            symbols_label, "Max reasoning iterations reached without a finalized decision; defaulting to sleep."
        )
        obs.update_trace(output={"action": forced.action, "regime": "unknown", "reasoning": forced.reasoning})
        _log(
            state,
            {
                "type": "decision",
                "action": forced.action,
                "status": forced.status,
                "price": forced.price,
                "quantity": forced.filled_quantity,
                "reasoning": forced.reasoning,
                "regime": "unknown",
            },
        )

    return "decided"


def _wait_for_next_cycle(state: "AppState", stop_event: threading.Event, cycle_sec: int) -> None:
    """Block until the next cycle is actually due.

    With no active alert or armed tactics, this is a plain `cycle_sec` timer
    (woken early only by fresh news). With an active alert or armed tactics,
    the fixed timer is disabled -- the agent committed to "nothing changes
    until a watched condition fires, a tactic executes, or news arrives", so it
    should wait indefinitely for `state.agent_wake_event` rather than also
    waking on the next scheduled tick. The price/news stream threads and the
    TacticsExecutor set that event directly the moment a condition is met, a
    conditional trade fills, or fresh news arrives -- never on a timer just to
    check state.
    """
    state.agent_wake_event.clear()
    state.agent_wake_reason = None
    if stop_event.is_set():
        return

    # The stream only signals on the *next* tick, so an alert condition that's
    # already satisfied the instant it's set would otherwise wait for a tick
    # that may not come. Catch that once, up front.
    alert_pairs = state.iter_alerts()
    if alert_pairs:
        hit = next((a for ss, a in alert_pairs if alert_triggered(ss, a)), None)
        if hit is not None:
            state.clear_alerts()
            _log(
                state,
                {"type": "status", "text": f"Alert already met: {format_alert(hit)}; waking early."},
            )
            return

    # An active alert or armed tactics (on any symbol) mean the agent should
    # sleep until a condition fires, a tactic executes, or news arrives -- not
    # get woken by the regular cycle timer too. (Armed tactics are watched
    # independently by the per-symbol TacticsExecutors, which wake this thread
    # on execution.)
    timeout = None if (alert_pairs or state.any_tactics()) else cycle_sec
    woke_early = state.agent_wake_event.wait(timeout=timeout)
    if stop_event.is_set():
        return
    if woke_early and state.agent_wake_reason:
        _log(state, {"type": "status", "text": f"{state.agent_wake_reason} Waking early."})
    state.agent_wake_event.clear()
    state.agent_wake_reason = None


def _wait_for_premarket_window(state: "AppState", stop_event: threading.Event) -> bool:
    """Block until PREMARKET_LEAD_SEC before the next opening bell -- the
    earliest moment the Premarket Analyst is allowed to start its analysis.
    Returns False when the agent was stopped while holding."""
    logged = False
    while not stop_event.is_set():
        remaining = market_hours.seconds_until_next_open() - PREMARKET_LEAD_SEC
        if remaining <= 0:
            return True
        if not logged:
            open_at = market_hours.next_market_open()
            _log(
                state,
                {
                    "type": "status",
                    "text": (
                        f"Premarket analyst holding until {PREMARKET_LEAD_SEC // 60} min "
                        f"before the bell (opens {open_at.strftime('%Y-%m-%d %H:%M UTC')})."
                    ),
                },
            )
            logged = True
        stop_event.wait(min(remaining, PREMARKET_WAIT_POLL_SEC))
    return False


def run_premarket_session(
    client: Any,
    model: str,
    symbols: list[str],
    state: "AppState",
    tracker: "DecisionTracker",
    stop_event: threading.Event,
) -> str:
    """Run the Premarket Analyst end to end: hold until PREMARKET_LEAD_SEC
    before the opening bell, run one opening-tactics cycle, then sleep until an
    armed tactic executes (the opening trade is simulated by the
    TacticsExecutor). Fresh news before the bell wakes it to revise the plan;
    any wake after the open just keeps it sleeping until a tactic fires.

    Returns "executed" once an opening tactic filled, "done" when the bell rang
    with nothing armed (nothing to perform), or "stopped".
    """
    while not stop_event.is_set():
        if not _wait_for_premarket_window(state, stop_event):
            return "stopped"

        state.agent_wake_event.clear()
        state.agent_wake_reason = None
        try:
            run_agent_cycle(
                client, model, symbols, state, tracker, personality=PREMARKET_PERSONALITY
            )
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Premarket cycle failed: {exc}"})

        if not state.any_tactics():
            # No opening plan -- hold through the bell (so a caller that
            # re-assesses on return doesn't spin pre-open) and retire.
            _log(
                state,
                {
                    "type": "status",
                    "text": "Premarket analyst armed no opening tactics; retiring at the bell.",
                },
            )
            while not stop_event.is_set() and not market_hours.is_market_open():
                stop_event.wait(PREMARKET_WAIT_POLL_SEC)
            return "stopped" if stop_event.is_set() else "done"

        # Opening tactics armed: sleep until the executor performs one.
        while not stop_event.is_set():
            state.agent_wake_event.wait()
            if stop_event.is_set():
                return "stopped"
            reason = state.agent_wake_reason or ""
            state.agent_wake_event.clear()
            state.agent_wake_reason = None
            if reason.startswith("Tactics executed"):
                _log(
                    state,
                    {
                        "type": "status",
                        "text": "Opening tactic executed; premarket analyst retiring.",
                    },
                )
                return "executed"
            if not state.any_tactics():
                # Tactics were cleared without an execution (external cancel).
                return "done"
            if not market_hours.is_market_open():
                # Pre-bell wake (fresh news / alert): revise the opening plan.
                _log(
                    state,
                    {
                        "type": "status",
                        "text": f"{reason} Premarket analyst revising the opening plan.",
                    },
                )
                break
            # Post-open wake that wasn't an execution: the bracket is still
            # armed and watched -- keep sleeping until a tactic fires.
    return "stopped"


def _premarket_loop(
    state: "AppState",
    tracker: "DecisionTracker",
    symbols: list[str],
    provider: str,
    api_key: str,
    model: str,
    stop_event: threading.Event,
) -> None:
    """Standalone Premarket Analyst run: one premarket session, then the agent
    disables itself -- the opening tactics were performed (or there was nothing
    to perform) and this personality never trades the session that follows."""
    client = get_agent_client(provider, api_key)
    outcome = run_premarket_session(client, model, symbols, state, tracker, stop_event)
    if outcome != "stopped" and state.agent_stop_event is stop_event:
        stop_agent(state)
    scoring.end_session(state, tracker)
    state.agent_running = False
    _log(state, {"type": "status", "text": "Premarket analyst disabled."})
    obs.flush()


def _agent_loop(
    state: "AppState",
    tracker: "DecisionTracker",
    symbols: list[str],
    provider: str,
    api_key: str,
    model: str,
    cycle_sec: int,
    stop_event: threading.Event,
    personality: str = DEFAULT_PERSONALITY,
) -> None:
    client = get_agent_client(provider, api_key)
    while not stop_event.is_set():
        try:
            run_agent_cycle(client, model, symbols, state, tracker, personality=personality)
        except Exception as exc:
            _log(state, {"type": "error", "text": f"Agent cycle failed: {exc}"})
        # Daily scoring may come due mid-session on a long-running agent; the
        # check is one stat() call once the day is scored.
        scoring.maybe_score_day(state, tracker)
        _wait_for_next_cycle(state, stop_event, cycle_sec)
    scoring.end_session(state, tracker)
    state.agent_running = False
    _log(state, {"type": "status", "text": "Agent stopped"})


def start_tactics_executor(state: "AppState", tracker: "DecisionTracker") -> None:
    """Start one background matcher per streamed symbol for its armed tactics;
    stopped by `stop_agent`. Shared by `launch_agent` and the Automatic
    orchestrator."""
    for sym_state in state.iter_symbol_states():
        executor = TacticsExecutor(sym_state, tracker)
        sym_state.tactics_executor = executor
        executor.start()


def launch_agent(
    state: "AppState",
    tracker: "DecisionTracker",
    symbols: list[str],
    api_key: str,
    provider: str = "openai",
    model: "str | None" = None,
    cycle_sec: int = 60,
    personality: str = DEFAULT_PERSONALITY,
) -> None:
    """Stop any running agent for this state, then start a new background cycle
    loop trading the whole symbol basket."""
    model = model or DEFAULT_AGENT_MODELS[provider]
    stop_agent(state)
    stop_event = threading.Event()
    state.agent_stop_event = stop_event
    state.agent_running = True
    scoring.begin_session(state, personality, symbols)
    start_tactics_executor(state, tracker)
    if personality == PREMARKET_PERSONALITY:
        # One-shot pre-open specialist: holds for the opening window, arms the
        # opening tactics, and disables itself once they execute.
        threading.Thread(
            target=_premarket_loop,
            args=(state, tracker, symbols, provider, api_key, model, stop_event),
            daemon=True,
        ).start()
        return
    threading.Thread(
        target=_agent_loop,
        args=(state, tracker, symbols, provider, api_key, model, cycle_sec, stop_event, personality),
        daemon=True,
    ).start()


def stop_agent(state: "AppState") -> None:
    if state.agent_stop_event:
        state.agent_stop_event.set()
    state.agent_running = False
    # Disarm every symbol's standing conditional orders and their matchers --
    # with no agent to wake, tactics must not keep trading on their own.
    for sym_state in state.iter_symbol_states():
        if sym_state.tactics_executor is not None:
            sym_state.tactics_executor.stop()
            sym_state.tactics_executor = None
        sym_state.tactics = None
    # Interrupt a blocked _wait_for_next_cycle immediately instead of letting
    # it sit until the timeout expires.
    state.agent_wake_event.set()
    # Push any buffered traces from the cycle(s) that just ran to Langfuse
    # before the background flusher would otherwise get to them.
    obs.flush()
