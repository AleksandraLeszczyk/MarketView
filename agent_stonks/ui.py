import base64
import html
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from . import market_hours
from .agent import (
    AGENT_PERSONALITIES,
    DEFAULT_PERSONALITY,
    PREMARKET_PERSONALITY,
    launch_agent,
    selectable_personalities,
    stop_agent,
)
from .automatic import AUTOMATIC_AVATAR, AUTOMATIC_KEY, AUTOMATIC_LABEL, launch_automatic
from .charts import (
    build_analysis_gauges,
    build_chart,
    build_gamma_chart,
    build_historical_chart,
    build_performance_chart,
    build_smart_money_chart,
    empty_chart,
)
from .config import (
    AGENT_CYCLE_SEC,
    AGENT_EQUITY_HISTORY_MAXLEN,
    AGENT_LOG_POLL_SEC,
    AGENT_PERFORMANCE_POLL_SEC,
    CHART_POLL_SEC,
    FEEDS,
    MAX_BARS,
    NEWS_IMPACT_COLORS,
    OPTIONS_POLL_SEC,
    OPTIONS_WALL_HISTORY_MAXLEN,
    PAPER_STARTING_CASH,
    PALETTE,
    POLL_SEC,
    SESSION_START,
    TIMEFRAMES,
    TRADE_FIXED_COST,
)
from .datalog import log_fetch, log_fetch_failure
from .decisions import DecisionTracker
from .historical import (
    HISTORICAL_PERIODS,
    SPY_SYMBOL,
    VIX_SYMBOL,
    estimate_dividend_return_10y,
    estimate_total_return,
    fetch_close_series,
    fetch_dividends,
    fetch_earnings_dates,
    fetch_market_indicators,
    fetch_price_target_history,
    fetch_smart_money_flow,
    fetch_static_analysis,
)
from .llm import DEFAULT_AGENT_MODELS, DEFAULT_NEWS_MODELS, ENV_KEYS, PROVIDERS, models_for
from .news import fetch_news_with_fallback, score_news_impacts
from .premarket import DEFAULT_PREMARKET_MODELS, PremarketBriefing, generate_premarket_analysis
from .options import fetch_options_walls_data
from .performance import compute_equity_curve, decision_markers, summarize
from .profile_model import predicted_open_profile
from .report import build_report_html
from .rest import fetch_bars, fetch_daily_bars, fetch_trades
from .state import (
    PRICE_AXIS_ALERT_FIELDS,
    AppState,
    SymbolState,
    current_volume_ratio,
    format_alert,
    format_tool_kv,
    momentum_pct,
    today_daily_bar,
)
from .tactics import tactic_price_levels, tactics_summaries
from .stream import backfill_bars, launch_stream, launch_stream_news
from .technical_analysis import (
    analyze_fair_value_gaps,
    analyze_intraday,
    analyze_liquidity,
    analyze_market,
    analyze_order_blocks,
    analyze_premium_discount,
    analyze_smart_money_setup,
    analyze_trend,
    get_put_call_walls_and_gamma,
)


def _get_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
    state = st.session_state["app_state"]
    # Streamlit's dev-mode autoreload reruns this script on every save but keeps
    # the same AppState instance alive in session_state. If a field was added to
    # AppState after this instance was constructed, the instance's __class__ (and
    # therefore __getattr__) still points at the pre-edit definition, so reading
    # the new field raises AttributeError instead of falling back to a default.
    # Writing straight into __dict__ sidesteps the class entirely.
    state.__dict__.setdefault("symbols", [])
    state.__dict__.setdefault("symbol_states", {})
    return state


def _parse_symbols(text: str) -> list[str]:
    """'aapl, tsla msft' -> ['AAPL', 'TSLA', 'MSFT'] (deduped, order kept)."""
    seen: list[str] = []
    for raw in re.split(r"[,;\s]+", text or ""):
        sym = raw.strip().upper()
        if sym and sym not in seen:
            seen.append(sym)
    return seen


def _effective_symbols(state: AppState, symbols_input: str) -> list[str]:
    """Symbols the panels should render: the sidebar input, falling back to
    whatever is currently streamed."""
    return _parse_symbols(symbols_input) or list(state.symbols)


def _personality_label(key: str) -> str:
    """Display label for a personality key, including the Automatic orchestrator."""
    if key == AUTOMATIC_KEY:
        return AUTOMATIC_LABEL
    entry = AGENT_PERSONALITIES.get(key) or AGENT_PERSONALITIES[DEFAULT_PERSONALITY]
    return entry["label"]


AVATAR_DIR = Path(__file__).resolve().parent.parent / "data" / "avatars"


@lru_cache(maxsize=None)
def _avatar_data_uri(key: str) -> Optional[str]:
    """Base64 data URI for a personality's avatar PNG, or None if the file is missing."""
    if key == AUTOMATIC_KEY:
        filename = AUTOMATIC_AVATAR
    else:
        entry = AGENT_PERSONALITIES.get(key) or AGENT_PERSONALITIES[DEFAULT_PERSONALITY]
        filename = entry["avatar"]
    try:
        data = (AVATAR_DIR / filename).read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _parse_ma_periods(ma_selection: list[str]) -> list[int]:
    mapping = {"VWMA(5)": 5, "VWMA(15)": 15, "VWMA(60)": 60}
    return [mapping[s] for s in ma_selection if s in mapping]


def _parse_avg_flags(ma_selection: list[str]) -> tuple[bool, bool, bool]:
    return "7d Avg" in ma_selection, "28d Avg" in ma_selection, "1y Avg" in ma_selection


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text)).strip()


def wrap_text(text: Optional[str], width: int = 80) -> str:
    """Insert <br> tags to wrap text at the given character width."""
    if not text:
        return ""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + (1 if current else 0) > width and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


_IMPACT_STYLE: dict[str, dict[str, str]] = {
    "positive": {"label": "positive impact", "dot": NEWS_IMPACT_COLORS["positive"], "bg": "#0d2b24", "border": "#1a4a3d", "text": "#26c6a2"},
    "negative": {"label": "negative impact", "dot": NEWS_IMPACT_COLORS["negative"], "bg": "#2b0d0d", "border": "#4a1a1a", "text": "#ef5350"},
    "neutral":  {"label": "neutral impact",  "dot": NEWS_IMPACT_COLORS["neutral"],  "bg": "#1e1e2e", "border": "#2a2d3a", "text": "#888888"},
    "small":    {"label": "small impact",    "dot": NEWS_IMPACT_COLORS["small"],    "bg": "#2b1a0d", "border": "#4a2d1a", "text": "#fb923c"},
    "unknown":  {"label": "unknown impact",  "dot": NEWS_IMPACT_COLORS["unknown"],  "bg": "#1a1d27", "border": "#2a2d3a", "text": "#555555"},
}


def _impact_badge(impact: str) -> str:
    cfg = _IMPACT_STYLE.get(impact, _IMPACT_STYLE["unknown"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'padding:3px 9px;border-radius:12px;background:{cfg["bg"]};'
        f'border:1px solid {cfg["border"]};font-size:10px;font-weight:600;'
        f'color:{cfg["text"]};white-space:nowrap;letter-spacing:0.02em;">'
        f'<span style="width:6px;height:6px;border-radius:50%;'
        f'background:{cfg["dot"]};display:inline-block;flex-shrink:0;"></span>'
        f'{cfg["label"]}</span>'
    )


def _news_html(news: list[dict], symbol: str, impacts: Optional[dict] = None) -> str:
    if not news:
        return (
            f"<p style='color:{PALETTE['muted']};padding:12px'>"
            f"No recent news for {symbol}.</p>"
        )
    impacts = impacts or {}
    cards = []
    for item in news[:12]:
        ts = pd.to_datetime(item.get("created_at")).strftime("%b %d  %H:%M")
        src = html.escape(item.get("source", ""))
        headline = html.escape(_strip_html(item.get("headline", "")))
        summary = _strip_html(item.get("summary") or "")[:180].rstrip()
        summary = html.escape(summary)
        url = html.escape(item.get("url", "#"))
        news_id = str(item.get("id", ""))
        impact = impacts.get(news_id)
        badge = _impact_badge(impact) if impact else _impact_badge("unknown")
        cards.append(
            f"""
        <div style="background:{PALETTE['panel']}; border-radius:8px; padding:12px 14px;
                    border:1px solid {PALETTE['grid']}; display:flex; flex-direction:column;
                    gap:6px; min-width:0;">
          <div style="font-size:11px; color:{PALETTE['muted']}; display:flex; align-items:center;
                      justify-content:space-between; gap:8px; flex-wrap:wrap;">
            <span>{ts} · <span style="color:{PALETTE['accent']}">{src}</span></span>
            {badge}
          </div>
          <a href="{url}" target="_blank"
             style="color:{PALETTE['text']}; font-weight:600;
                    text-decoration:none; font-size:13px; line-height:1.4">
            {headline}
          </a>
          <div style="font-size:12px; color:{PALETTE['muted']}; line-height:1.5">
            {summary}{"…" if summary else ""}
          </div>
        </div>"""
        )
    return f"""
    <div style="font-family:Inter,sans-serif; padding:4px 0 12px;">
      <h3 style="color:{PALETTE['text']}; font-size:14px; margin:0 0 10px 0">
        📰 Latest news · <b style="color:{PALETTE['accent']}">{symbol}</b>
      </h3>
      <div style="display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
                  gap:10px;">
        {''.join(cards)}
      </div>
    </div>"""


build_news_html = _news_html


def _today_range(
    daily_bars: list[dict], intraday_bars: list[dict]
) -> tuple[float | None, float | None, float | None]:
    """(low, high, open) for today's session.

    Prefers today's still-forming daily bar; falls back to today's intraday bars
    (for pre-open/lagging feeds where the daily bar hasn't published yet)."""
    bar = today_daily_bar(daily_bars)
    if bar is not None:
        try:
            return float(bar["l"]), float(bar["h"]), float(bar["o"])
        except (KeyError, TypeError, ValueError):
            pass

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    todays = [b for b in intraday_bars if str(b.get("t", "")).startswith(today)]
    if not todays:
        return None, None, None
    try:
        low = min(float(b["l"]) for b in todays)
        high = max(float(b["h"]) for b in todays)
        open_ = float(todays[0]["o"])
    except (KeyError, TypeError, ValueError):
        return None, None, None
    return low, high, open_


def _quote_html(
    price: float | None,
    prev_close: float | None,
    bid: float | None,
    bid_size: float | None,
    ask: float | None,
    ask_size: float | None,
    previous_minute_high: float | None,
    previous_minute_low: float | None,
    symbol: str,
    today_low: float | None = None,
    today_high: float | None = None,
    current_momentum: float | None = None,
    daily_momentum: float | None = None,
) -> str:
    if price is None and bid is None and ask is None:
        return ""
    change = price - prev_close if price is not None and prev_close else None
    pct = (change / prev_close * 100) if change is not None and prev_close else None
    if change is None:
        arrow, chg_color, delta_str = "", PALETTE["muted"], ""
    elif change >= 0:
        arrow, chg_color = "▲", "#26c6a2"
        delta_str = f"+{change:.2f} ({pct:+.2f}%)"
    else:
        arrow, chg_color = "▼", "#ef5350"
        delta_str = f"{change:.2f} ({pct:.2f}%)"

    symbol_chip = (
        f'<span style="font-size:13px;font-weight:700;color:{PALETTE["accent"]};'
        f'letter-spacing:0.04em;margin-right:10px;">{html.escape(symbol)}</span>'
    )
    price_row = ""
    if price is not None:
        price_row = (
            f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;">'
            f'{symbol_chip}'
            f'<span style="font-size:28px;font-weight:700;color:{PALETTE["text"]};'
            f'letter-spacing:-0.5px;">${price:,.4f}</span>'
            f'<span style="font-size:14px;font-weight:600;color:{chg_color};">'
            f'{arrow} {delta_str}</span>'
            f'</div>'
        )
    else:
        price_row = f'<div style="margin-bottom:6px;">{symbol_chip}</div>'

    def _side(label: str, p: float | None, sz: float | None, color: str) -> str:
        if p is None:
            return ""
        size_str = f'<span style="font-size:11px;color:{PALETTE["muted"]};margin-left:4px;">{sz:,.0f}</span>' if sz else ""
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:{PALETTE["panel"]};border:1px solid {PALETTE["grid"]};'
            f'border-radius:8px;padding:8px 16px;min-width:100px;">'
            f'<span style="font-size:10px;font-weight:600;color:{PALETTE["muted"]};'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">{label}</span>'
            f'<span style="font-size:18px;font-weight:700;color:{color};">${p:,.4f}</span>'
            f'{size_str}'
            f'</div>'
        )

    low_card = _side("Prev Min Low", previous_minute_low, None, PALETTE["muted"])
    bid_card = _side("Bid", bid, bid_size, "#ef5350")
    ask_card = _side("Ask", ask, ask_size, "#26c6a2")
    high_card = _side("Prev Min High", previous_minute_high, None, PALETTE["muted"])
    spread_row = ""
    if bid is not None and ask is not None:
        spread = ask - bid
        spread_row = (
            f'<span style="font-size:11px;color:{PALETTE["muted"]};align-self:center;">'
            f'spread {spread:.4f}</span>'
        )

    ba_row = ""
    if bid_card or ask_card or low_card or high_card:
        ba_row = (
            f'<div style="display:flex;gap:10px;align-items:stretch;">'
            f'{low_card}{bid_card}{spread_row}{ask_card}{high_card}'
            f'</div>'
        )

    def _stat_card(label: str, value: str, color: str) -> str:
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:{PALETTE["panel"]};border:1px solid {PALETTE["grid"]};'
            f'border-radius:8px;padding:8px 16px;min-width:100px;">'
            f'<span style="font-size:10px;font-weight:600;color:{PALETTE["muted"]};'
            f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">{label}</span>'
            f'<span style="font-size:18px;font-weight:700;color:{color};">{value}</span>'
            f'</div>'
        )

    def _momentum_color(m: float | None) -> str:
        if m is None or abs(m) < 1e-9:
            return PALETTE["muted"]
        return "#26c6a2" if m > 0 else "#ef5350"

    day_low_card = (
        _stat_card("Day Low", f"${today_low:,.4f}", PALETTE["muted"])
        if today_low is not None else ""
    )
    day_high_card = (
        _stat_card("Day High", f"${today_high:,.4f}", PALETTE["muted"])
        if today_high is not None else ""
    )
    cur_mom_card = (
        _stat_card("Momentum (10m)", f"{current_momentum:+.2f}%", _momentum_color(current_momentum))
        if current_momentum is not None else ""
    )
    day_mom_card = (
        _stat_card("Momentum (day)", f"{daily_momentum:+.2f}%", _momentum_color(daily_momentum))
        if daily_momentum is not None else ""
    )

    stats_row = ""
    if day_low_card or day_high_card or cur_mom_card or day_mom_card:
        stats_row = (
            f'<div style="display:flex;gap:10px;align-items:stretch;margin-top:10px;">'
            f'{day_low_card}{day_high_card}{cur_mom_card}{day_mom_card}'
            f'</div>'
        )

    return (
        f'<div style="font-family:Inter,monospace;padding:4px 0 8px;">'
        f'{price_row}{ba_row}{stats_row}'
        f'</div>'
    )


def _volume_alert_banner(state: AppState, sym_state: SymbolState) -> None:
    """Live relative-volume readout + a prominent banner once the alert fires."""
    if not state.volume_alert_enabled:
        return
    with sym_state.lock:
        day_volume = sym_state.day_volume
        triggered = sym_state.volume_alert_triggered
    multiplier = state.volume_alert_multiplier
    daily_bars = sym_state.daily_bars
    ratio, _ = current_volume_ratio(day_volume, daily_bars)
    if ratio is None:
        return
    if triggered:
        st.html(
            f"<div style='background:{PALETTE['orange']};color:#1a1d27;"
            "font-family:Inter,sans-serif;font-weight:600;border-radius:6px;"
            "padding:8px 12px;margin:4px 0'>"
            f"⚡ {sym_state.symbol} high volume: {ratio:.2f}× average daily volume "
            f"(alert threshold {multiplier:.1f}×)</div>"
        )
    else:
        st.caption(
            f"📊 {sym_state.symbol} relative volume: {ratio:.2f}× avg daily volume "
            f"(alerts at {multiplier:.1f}×)"
        )


@st.fragment(run_every=POLL_SEC)
def _price_ticker() -> None:
    state = _get_state()
    st.caption(f"Status: {state.status}")
    if state.news_status not in ("Idle", state.status):
        st.caption(f"News: {state.news_status}")
    for sym_state in state.iter_symbol_states():
        with sym_state.lock:
            last_price = sym_state.last_price
            prev_close = sym_state.prev_close
            bid_price = sym_state.bid_price
            bid_size = sym_state.bid_size
            ask_price = sym_state.ask_price
            ask_size = sym_state.ask_size
            previous_minute_high = sym_state.previous_minute_high
            previous_minute_low = sym_state.previous_minute_low
            bars = list(sym_state.bars)
        today_low, today_high, today_open = _today_range(sym_state.daily_bars, bars)
        current_momentum = momentum_pct(sym_state)
        daily_momentum = (
            (last_price / today_open - 1.0) * 100.0
            if last_price is not None and today_open
            else None
        )
        quote = _quote_html(
            last_price, prev_close, bid_price, bid_size, ask_price, ask_size,
            previous_minute_high, previous_minute_low, sym_state.symbol,
            today_low=today_low, today_high=today_high,
            current_momentum=current_momentum, daily_momentum=daily_momentum,
        )
        if quote:
            st.html(quote)
        _volume_alert_banner(state, sym_state)


@st.fragment(run_every=CHART_POLL_SEC)
def _chart_panel() -> None:
    state = _get_state()
    tracker = state.decision_tracker
    rendered = False
    for sym_state in state.iter_symbol_states():
        sym = sym_state.symbol
        with sym_state.lock:
            bars = list(sym_state.bars)
            # Only price-axis alerts (price/bid/ask/day high/low) can be drawn as
            # horizontal lines; volume/spread/portfolio alerts have no price level.
            price_alerts = [
                a for a in sym_state.alerts if a.get("field") in PRICE_AXIS_ALERT_FIELDS
            ]

        if not bars:
            continue
        rendered = True

        # Price levels at which armed tactics (standing conditional orders) execute.
        tactic_levels = tactic_price_levels(sym_state.tactics)
        decisions = tracker.trade_markers(symbol=sym) if tracker else None

        predicted_profile = (
            predicted_open_profile(sym_state, bars)
            if state.show_predicted_profile
            else None
        )

        fig = build_chart(
            bars,
            sym_state.news,
            sym_state.trades,
            sym,
            SESSION_START,
            ma_periods=state.ma_periods,
            show_fib=state.show_fib,
            show_7d_avg=state.show_7d_avg,
            show_28d_avg=state.show_28d_avg,
            show_1y_avg=state.show_1y_avg,
            mixture_distribution=state.mixture_distribution,
            mixture_max_components=state.mixture_max_components,
            predicted_profile=predicted_profile,
            mixture_fit_target=state.mixture_fit_target,
            daily_bars=sym_state.daily_bars,
            vwap_style=state.vwap_style,
            show_candle_body=state.show_candle_body,
            show_percentile_body=state.show_percentile_body,
            show_whiskers=state.show_whiskers,
            decisions=decisions,
            price_alerts=price_alerts,
            tactic_levels=tactic_levels,
            news_impacts=sym_state.news_impacts,
            fill_gaps=state.fill_gaps,
        )
        st.plotly_chart(fig, width='stretch', key=f"live_chart_{sym}")
    if not rendered:
        st.plotly_chart(empty_chart(), width='stretch', key="live_chart_empty")


def _live_chart_controls() -> None:
    state = _get_state()
    with st.expander("Chart Settings"):
        st.selectbox("Timeframe", TIMEFRAMES, index=0, key="live_timeframe")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Candle**")
            show_candle_body = st.checkbox("Open-Close", value=True)
            show_percentile_body = st.checkbox("20%-80%", value=False)
            show_whiskers = st.checkbox("Whiskers", value=True)
            fill_gaps = st.checkbox(
                "Fill no-trade gaps",
                value=True,
                help="Draw flat zero-volume placeholder bars for feed minutes "
                "without any trade (common on IEX for thin symbols).",
            )
            vwap_style = st.selectbox("VWAP", ["hide", "dot", "line"], index=0)
        with c2:
            st.markdown("**Overlays**")
            vwma_selection = st.multiselect(
                "VWMA",
                ["VWMA(5)", "VWMA(15)", "VWMA(60)"],
                default=[],
            )
            avg_selection = st.multiselect(
                "Average Lines",
                ["7d Avg", "28d Avg", "1y Avg"],
                default=[],
            )
            show_fib = st.checkbox("Fibonacci levels", value=False)

        st.markdown("**Price Profile Fit**")
        dist_choice = st.selectbox("Fit mixture", ["None", "Gaussian", "Cauchy"], index=0)
        fit_enabled = dist_choice != "None"
        max_components = st.slider(
            "Components",
            min_value=1,
            max_value=5,
            value=1,
            disabled=not fit_enabled,
        )
        show_predicted = st.checkbox(
            "ML predicted profile",
            value=False,
            help="Draws where today's volume is predicted to trade (LevelsML "
            "density model: per-quantile LightGBM at the open, from daily-bar "
            "features). Needs the trained pack in ../Models and daily bars.",
        )
        fit_target_choice = st.selectbox(
            "Fit to",
            ["Live volume", "Predicted profile"],
            index=0,
            disabled=not (fit_enabled and show_predicted),
            help="Which profile the Gaussian/Cauchy mixture is fitted to.",
        )

        st.markdown("**Data**")
        backfill_clicked = st.button(
            "⟲ Backfill missing bars",
            disabled=not (state.symbols and state.api_key),
            help="Re-fetch each symbol's session bars via REST (yfinance if that "
            "fails) and merge any that the stream missed, e.g. during a reconnect.",
        )
        if backfill_clicked:
            with st.spinner("Backfilling…"):
                notes = []
                for sym_state in state.iter_symbol_states():
                    try:
                        added, source = backfill_bars(
                            sym_state.symbol, state.api_key, state.api_secret,
                            state.feed, sym_state, state.timeframe,
                        )
                    except Exception as exc:
                        st.error(f"Backfill failed for {sym_state.symbol}: {exc}")
                    else:
                        notes.append(
                            f"{sym_state.symbol}: {added} bar(s) via {source}"
                            if added else f"{sym_state.symbol}: no missing bars"
                        )
                if notes:
                    st.caption(" · ".join(notes))

    state.ma_periods = _parse_ma_periods(vwma_selection)
    state.show_7d_avg, state.show_28d_avg, state.show_1y_avg = _parse_avg_flags(avg_selection)
    state.show_candle_body = show_candle_body
    state.show_percentile_body = show_percentile_body
    state.show_whiskers = show_whiskers
    state.fill_gaps = fill_gaps
    state.vwap_style = vwap_style
    state.show_fib = show_fib
    state.mixture_distribution = dist_choice.lower() if fit_enabled else "none"
    state.mixture_max_components = max_components if fit_enabled else 0
    state.show_predicted_profile = show_predicted
    state.mixture_fit_target = (
        "predicted"
        if show_predicted and fit_target_choice == "Predicted profile"
        else "live"
    )


def _volume_alert_controls() -> None:
    state = _get_state()
    with st.expander("🔔 Volume Alert"):
        st.caption(
            "Alerts when any symbol's cumulative volume today exceeds a multiple of "
            "its average daily volume (mean of the last 20 completed days; yesterday's "
            "volume early on). Wakes the agent early when it fires. On by default."
        )
        c1, c2 = st.columns([1, 1.4])
        enabled = c1.checkbox("Enabled", value=state.volume_alert_enabled, key="vol_alert_enabled")
        multiplier = c2.number_input(
            "× avg daily volume",
            min_value=0.1,
            value=float(state.volume_alert_multiplier),
            step=0.1,
            format="%.1f",
            key="vol_alert_multiplier",
        )
    # Changing the threshold or re-enabling clears the one-shot latches so the
    # alert can fire again under the new settings.
    if enabled != state.volume_alert_enabled or multiplier != state.volume_alert_multiplier:
        for sym_state in state.iter_symbol_states():
            sym_state.volume_alert_triggered = False
            sym_state.volume_alert_ratio = None
    state.volume_alert_enabled = enabled
    state.volume_alert_multiplier = multiplier


def _news_analysis_controls(symbols: list[str]) -> None:
    state = _get_state()
    provider = st.selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(state.news_llm_provider),
        key="news_llm_provider_select",
        help=f"Model used: {', '.join(f'{p}={m}' for p, m in DEFAULT_NEWS_MODELS.items())}",
    )
    state.news_llm_provider = provider
    env_var = ENV_KEYS[provider]
    llm_key = os.getenv(env_var, "")
    if not llm_key:
        st.caption(f"⚠️ {env_var} is not set.")

    analyze_clicked = st.button("🔍 Analyze News", key="news_analyze_btn")
    if analyze_clicked:
        states = [s for s in state.iter_symbol_states() if s.symbol in symbols] or list(
            state.iter_symbol_states()
        )
        if not any(s.news for s in states):
            st.warning("No news loaded yet. Start the Live stream for the symbols first.")
        elif not llm_key:
            st.error(f"{env_var} is not set; news analysis needs an LLM key.")
        else:
            with st.spinner("Scoring news impact…"):
                for sym_state in states:
                    if not sym_state.news:
                        continue
                    try:
                        sym_state.news_impacts = score_news_impacts(
                            sym_state.symbol, sym_state.news, provider, llm_key
                        )
                    except Exception as exc:
                        st.error(f"News impact scoring failed for {sym_state.symbol}: {exc}")


@st.fragment(run_every=CHART_POLL_SEC)
def _news_panel(symbols: list[str]) -> None:
    state = _get_state()
    if state.news_status not in ("Idle", state.status):
        st.caption(f"News: {state.news_status}")
    _news_analysis_controls(symbols)
    rendered = False
    for sym_state in state.iter_symbol_states():
        st.html(_news_html(sym_state.news, sym_state.symbol, sym_state.news_impacts))
        rendered = True
    if not rendered:
        st.info("Start the Live stream to load news for your symbols.")


def _live_panel() -> None:
    _live_chart_controls()
    # _volume_alert_controls()
    _price_ticker()
    _chart_panel()


def _historical_panel(symbols: list[str]) -> None:
    period_label = st.selectbox("Period", list(HISTORICAL_PERIODS.keys()), index=3, key="hist_period")

    if not symbols:
        st.plotly_chart(empty_chart("Enter symbols in the sidebar"), width='stretch')
        return

    # Targets default to shown: select symbols the first time they appear in the
    # basket (keeping the user's deselections for ones they've already seen), and
    # drop selections for symbols no longer in the basket before the widget renders.
    seen_syms = st.session_state.setdefault("hist_target_seen", set())
    selected = st.session_state.get("hist_target_syms") or []
    st.session_state["hist_target_syms"] = [s for s in selected if s in symbols] + [
        s for s in symbols if s not in seen_syms and s not in selected
    ]
    seen_syms.update(symbols)
    target_syms = st.pills(
        "Expert price targets",
        symbols,
        selection_mode="multi",
        key="hist_target_syms",
        help=(
            "Overlay each analyst firm's price targets as piecewise lines over the shown "
            "period (Yahoo's analyst actions feed). Click a firm in a chart's legend to "
            "hide that firm's line."
        ),
    ) or []

    days = HISTORICAL_PERIODS[period_label]
    # Shared context series fetched once for the whole basket.
    try:
        spy_close = fetch_close_series(SPY_SYMBOL, days)
        vix_close = fetch_close_series(VIX_SYMBOL, days)
    except Exception as exc:
        st.error(f"Failed to load market context series: {exc}")
        spy_close, vix_close = None, None

    for sym in symbols:
        with st.spinner(f"Loading historical data for {sym}…"):
            try:
                ticker_close = fetch_close_series(sym, days)
                dividends = fetch_dividends(sym, days)
                earnings = fetch_earnings_dates(sym, days)
            except Exception as exc:
                st.error(f"Failed to load historical data for {sym}: {exc}")
                continue
            price_targets = fetch_price_target_history(sym, days) if sym in target_syms else None

        fig = build_historical_chart(
            ticker_close,
            spy_close if sym != SPY_SYMBOL else None,
            vix_close,
            sym,
            period_label,
            dividends,
            earnings,
            price_targets=price_targets,
        )
        st.plotly_chart(fig, width='stretch', key=f"hist_chart_{sym}")
        _static_analysis_panel(sym)


def _static_analysis_panel(symbol: str) -> None:
    static = fetch_static_analysis(symbol)
    pe_ratio = static["pe_ratio"]
    dividend_yield = static["dividend_yield"]
    growth_rate = static["growth_rate"]
    total_return = estimate_total_return(dividend_yield, growth_rate)
    dividend_return_10y = estimate_dividend_return_10y(dividend_yield, growth_rate)

    col1, col2, col3 = st.columns(3)
    col1.metric("P/E (trailing)", f"{pe_ratio:.2f}" if pe_ratio is not None else "—")
    col2.metric(
        "Est. annual return (growth + div)",
        f"{total_return * 100:.1f}%" if total_return is not None else "—",
    )
    col3.metric(
        "Est. 10yr cumulative dividend return",
        f"{dividend_return_10y * 100:.1f}%" if dividend_return_10y is not None else "—",
    )


@st.fragment(run_every=CHART_POLL_SEC)
def _technical_analysis_panel(symbols: list[str]) -> None:
    """Visualizes the three human-readable reads from `technical_analysis` for
    each symbol: the daily trend regime, intraday momentum, and (once, shared)
    the broad market environment."""
    state = _get_state()

    states = [s for s in state.iter_symbol_states() if s.symbol in symbols] or list(
        state.iter_symbol_states()
    )
    states = [s for s in states if s.bars]
    if not states:
        st.info("Start the Live stream for your symbols first so there's data to analyze.")
        return

    # The broad-market read is symbol-independent -- compute it once.
    try:
        market_series = fetch_market_indicators()
        market = analyze_market(market_series.get("vix"), market_series.get("spy"), market_series.get("vix3m"))
    except Exception as exc:
        market = {"note": f"market indicators unavailable: {exc}"}

    for sym_state in states:
        sym = sym_state.symbol
        with sym_state.lock:
            bars = list(sym_state.bars)
        daily_bars = sym_state.daily_bars

        st.subheader(sym)
        trend = analyze_trend(daily_bars if daily_bars else bars)
        intraday = analyze_intraday(bars)

        st.plotly_chart(
            build_analysis_gauges(trend, intraday, market),
            width='stretch',
            key=f"analysis_gauges_{sym}",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📈 Trend (daily)**")
            if "note" in trend:
                st.caption(trend["note"])
            else:
                st.metric(
                    "Regime",
                    f"{trend['regime'].capitalize()} ({trend['trend_strength']})",
                    f"{trend['pct_change_over_period']:+.1f}%",
                )
                st.caption(trend["summary"])
        with c2:
            st.markdown("**⚡ Intraday Momentum**")
            if "note" in intraday:
                st.caption(intraday["note"])
            else:
                st.metric("Window change", f"{intraday['pct_change_in_window']:+.2f}%", intraday["momentum_pattern"])
                st.caption(intraday["summary"])
        with c3:
            st.markdown("**🌍 Market Environment**")
            if "note" in market:
                st.caption(market["note"])
            else:
                st.metric("Risk environment", market["risk_environment"].capitalize(), f"score {market['risk_score']:+d}")
                st.caption(market["summary"])
                for insight in market.get("insights", []):
                    st.markdown(f"- {insight}")
        st.divider()


@st.fragment(run_every=CHART_POLL_SEC)
def _smart_money_panel(symbols: list[str]) -> None:
    """Visualizes the Smart Money Concepts setup per symbol: higher-timeframe daily
    order blocks drawn as demand/supply zones over the candles, with the suggested
    entry/stop/target geometry -- the same composite read the `smart_money` agent
    personality trades from, with the intraday confirmation reported alongside."""
    state = _get_state()

    states = [s for s in state.iter_symbol_states() if s.symbol in symbols] or list(
        state.iter_symbol_states()
    )
    states = [s for s in states if s.daily_bars]
    if not states:
        st.info("Start the Live stream for your symbols first so there's daily structure to analyze.")
        return

    for sym_state in states:
        sym = sym_state.symbol
        with sym_state.lock:
            intraday = list(sym_state.bars)
            spot = sym_state.last_price
        daily = sym_state.daily_bars

        st.subheader(sym)
        setup = analyze_smart_money_setup(daily, intraday_bars=intraday, spot=spot)
        blocks = analyze_order_blocks(daily, spot=spot).get("order_blocks", [])
        pd_read = analyze_premium_discount(daily, spot=spot)
        liquidity = analyze_liquidity(intraday, spot=spot) if intraday else {}

        # The chart renders daily candles, so only the daily order blocks (whose indices
        # map to that x-axis) are drawn as zones. Intraday FVGs drive the agent's
        # confirmation read but have no position on a daily chart, so they're left off.
        # Premium/discount (daily) and the nearest liquidity pools overlay as levels.
        chart_analysis = {
            **setup,
            "order_blocks": blocks,
            "fair_value_gaps": [],
            "premium_discount": pd_read,
            "liquidity": liquidity,
        }
        st.plotly_chart(
            build_smart_money_chart(daily, chart_analysis, sym),
            width="stretch",
            key=f"smart_money_chart_{sym}",
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", setup.get("signal", "n/a").replace("_", " "), setup.get("quality", ""))
        rr = setup.get("reward_risk_to_target")
        c2.metric("Reward : Risk", f"{rr:.1f}:1" if rr is not None else "n/a")
        c3.metric("Range zone", (setup.get("premium_discount_zone") or "n/a").title())
        confs = setup.get("intraday_confirmation") or []
        c4.metric("Intraday confirmation", ", ".join(confs) if confs else "none")

        st.caption(setup.get("summary", ""))

        sweep = setup.get("recent_sweep")
        if sweep is not None:
            st.markdown(
                f"- **Liquidity sweep:** {sweep['type']} stop-run of {sweep['level']:.2f} "
                f"({sweep['bars_ago']} bars ago)"
            )

        ob = setup.get("order_block")
        if ob is not None:
            e, s, t = setup.get("suggested_entry"), setup.get("suggested_stop"), setup.get("structural_target")
            st.markdown(
                f"- **Demand order block:** {ob['bottom']:.2f}–{ob['top']:.2f} "
                f"({'unmitigated' if not ob['mitigated'] else 'mitigated'}, {ob['bars_ago']} bars ago)"
            )
            if e is not None and s is not None and t is not None:
                st.markdown(f"- **Geometry:** entry {e:.2f} · stop {s:.2f} · target {t:.2f}")
        if "note" in setup and setup.get("order_block") is None:
            st.caption(setup["note"])

        with st.expander(f"🏦 Institutional footprint — {sym} (insiders · 13F)", expanded=False):
            try:
                flow = fetch_smart_money_flow(sym)
            except Exception:
                flow = None
            if not flow:
                st.caption("Institutional ownership data unavailable.")
            else:
                st.caption(flow.get("summary", ""))
                f1, f2, f3 = st.columns(3)
                inst_pct = flow.get("institutions_pct_held")
                ins_pct = flow.get("insiders_pct_held")
                f1.metric("Institutions", f"{inst_pct * 100:.1f}%" if inst_pct is not None else "n/a")
                f2.metric("Insiders", f"{ins_pct * 100:.1f}%" if ins_pct is not None else "n/a")
                insider = flow.get("insider_flow")
                f3.metric(
                    "Insider 6mo",
                    insider["direction"].title() if insider else "n/a",
                    f"{insider['net_shares_6mo']:+,}" if insider else None,
                )
                for h in (flow.get("top_institutional_holders") or [])[:5]:
                    chg = h.get("pct_change")
                    chg_str = f" ({chg * 100:+.1f}% q/q)" if chg is not None else ""
                    held = f"{h['pct_held'] * 100:.2f}%" if h.get("pct_held") is not None else "n/a"
                    st.markdown(f"- **{h['holder']}** — {held}{chg_str}")
        st.divider()


def _record_wall_snapshot(sym_state: SymbolState, call_wall: float, put_wall: float) -> list[dict]:
    """Append a {call_wall, put_wall} snapshot if it differs from the last one, so the
    agent's trend read (rising/falling walls) reflects real shifts, not poll noise."""
    with sym_state.lock:
        history = list(sym_state.options_wall_history)
        last = history[-1] if history else None
        if last is None or last.get("call_wall") != call_wall or last.get("put_wall") != put_wall:
            history.append(
                {
                    "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                    "call_wall": call_wall,
                    "put_wall": put_wall,
                }
            )
            history = history[-OPTIONS_WALL_HISTORY_MAXLEN:]
            sym_state.options_wall_history = history
        return history


@st.fragment(run_every=OPTIONS_POLL_SEC)
def _options_walls_panel(symbols: list[str]) -> None:
    """Independently fetches/refreshes each symbol's options chain (cached, on its
    own poll loop -- never triggered by the agent) and renders the Call Wall /
    Put Wall / gamma read. The agent's get_put_call_walls tool only reads whatever
    this last stored on the SymbolState."""
    state = _get_state()
    if not symbols:
        st.plotly_chart(empty_chart("Enter symbols in the sidebar"), width='stretch')
        return

    for sym in symbols:
        sym_state = state.sym(sym)
        st.subheader(sym)
        live_spot = None
        if sym_state is not None:
            with sym_state.lock:
                live_spot = sym_state.last_price

        try:
            data = fetch_options_walls_data(sym, spot=live_spot)
        except Exception as exc:
            data = None
            if sym_state is not None:
                with sym_state.lock:
                    data = sym_state.options_chain
            if not data:
                st.error(f"Failed to fetch options chain for {sym}: {exc}")
                continue
            st.warning(f"Using last successful options fetch for {sym} -- refresh failed: {exc}")
        else:
            if sym_state is not None:
                with sym_state.lock:
                    sym_state.options_chain = data

        prior_history: list[dict] = []
        if sym_state is not None:
            with sym_state.lock:
                prior_history = list(sym_state.options_wall_history)
        analysis = get_put_call_walls_and_gamma(
            strikes=data["strikes"],
            calls_oi=data["calls_oi"],
            puts_oi=data["puts_oi"],
            calls_gamma_exposure=data["calls_gamma_exposure"],
            puts_gamma_exposure=data["puts_gamma_exposure"],
            spot=data["spot"],
            wall_history=prior_history,
        )
        if sym_state is not None:
            _record_wall_snapshot(sym_state, analysis["call_wall"], analysis["put_wall"])

        st.caption(f"Expiry {data['expiry']} · fetched {data['fetched_at']}")
        fig = build_gamma_chart(data, analysis, sym)
        st.plotly_chart(fig, width='stretch', key=f"gamma_chart_{sym}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Call Wall (resistance)", f"${analysis['call_wall']:.2f}", analysis["call_wall_trend"] or "")
        c2.metric("Put Wall (support)", f"${analysis['put_wall']:.2f}", analysis["put_wall_trend"] or "")
        c3.metric("Net gamma regime", analysis["gamma_regime"].split(" ")[0].capitalize())

        st.caption(analysis["summary"])
        for insight in analysis["insights"]:
            st.markdown(f"- {insight}")
        st.divider()


def _agent_entry_style(entry: dict) -> tuple[str, str, str]:
    """Return (icon, accent_color, label) for an agent log entry."""
    etype = entry.get("type")
    symbol = entry.get("symbol")
    suffix = f" {symbol}" if symbol else ""
    if etype == "decision":
        action = entry.get("action")
        if action == "buy":
            return "🟢", PALETTE["up"], f"BUY{suffix}"
        if action == "sell":
            return "🔴", PALETTE["down"], f"SELL{suffix}"
        if action == "alert":
            return "⏰", PALETTE["accent"], "ALERT"
        return "💤", PALETTE["muted"], "SLEEP"
    if etype == "tactics_set":
        if entry.get("cancelled") is not None:
            return "🎯", PALETTE["muted"], f"TACTICS CANCELLED{suffix}"
        return "🎯", PALETTE["orange"], f"TACTICS SET{suffix}"
    if etype == "tactics_execution":
        action = entry.get("action")
        if action == "buy":
            return "🎯", PALETTE["up"], f"TACTICS → BUY{suffix}"
        if action == "sell":
            return "🎯", PALETTE["down"], f"TACTICS → SELL{suffix}"
        return "🎯", PALETTE["orange"], f"TACTICS{suffix}"
    if etype == "regime_select":
        return "🤖", PALETTE["orange"], "REGIME → STRATEGY"
    if etype == "stand_down":
        return "🛑", PALETTE["orange"], "STAND DOWN"
    if etype == "tool_call":
        return "🛠️", PALETTE["accent"], str(entry.get("name", "tool"))
    if etype == "analysis":
        return "🧠", PALETTE["text"], "analysis"
    if etype == "cycle_start":
        return "🔄", PALETTE["accent"], "cycle start"
    if etype == "news_alert":
        return "📰", PALETTE["accent"], "NEWS ALERT"
    if etype == "error":
        return "⚠️", PALETTE["down"], "error"
    return "ℹ️", PALETTE["muted"], "status"


def _kv_row_html(data: dict) -> str:
    """Render a tool call's args/result dict as compact monospace key=value
    chips instead of a raw (and often mid-token truncated) JSON blob."""
    pairs = " &nbsp;·&nbsp; ".join(
        f"<span style='color:{PALETTE['muted']}'>{html.escape(k)}</span>="
        f"<span>{html.escape(v)}</span>"
        for k, v in format_tool_kv(data)
    )
    return f"<span style='font-family:monospace;font-size:11px'>{pairs}</span>"


def _agent_entry_body(entry: dict) -> str:
    etype = entry.get("type")
    if etype == "decision":
        price = entry.get("price")
        price_str = f"${price:,.4f}" if price is not None else "—"
        qty = entry.get("quantity") or 0
        regime = html.escape(str(entry.get("regime", "unknown")))
        reasoning = html.escape(entry.get("reasoning", ""))
        extra = ""
        if entry.get("action") == "alert" and entry.get("alerts"):
            levels = " or ".join(
                f"<b>{html.escape(format_alert(a))}</b>" for a in entry["alerts"]
            )
            extra = f" · Wake when {levels}"
        return (
            f"<div>Regime: <b>{regime}</b> · Qty: <b>{qty:.2f}</b> · Price: <b>{price_str}</b>{extra}</div>"
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
        )
    if etype == "tactics_set":
        cancelled = entry.get("cancelled")
        reasoning = html.escape(entry.get("reasoning", ""))
        if cancelled is not None:
            what = " · ".join(html.escape(t) for t in cancelled) or "none armed"
            return (
                f"<div>Cancelled: {what}</div>"
                f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
            )
        rows = "".join(f"<div>🎯 <b>{html.escape(t)}</b></div>" for t in entry.get("tactics") or [])
        replaced = entry.get("replaced") or []
        replaced_html = (
            f"<div style='color:{PALETTE['muted']}'>replaced: {' · '.join(html.escape(t) for t in replaced)}</div>"
            if replaced
            else ""
        )
        return f"{rows}{replaced_html}<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
    if etype == "tactics_execution":
        price = entry.get("price")
        price_str = f"${price:,.4f}" if price is not None else "—"
        qty = entry.get("quantity") or 0
        status = html.escape(str(entry.get("status", "")))
        tactic = html.escape(entry.get("tactic", ""))
        triggered = html.escape(entry.get("triggered_by", ""))
        error = entry.get("error")
        error_html = (
            f"<div style='color:{PALETTE['down']}'>{html.escape(str(error))}</div>" if error else ""
        )
        return (
            f"<div>Executed <b>{tactic}</b> · Status: <b>{status}</b> · Qty: <b>{qty:.2f}</b> · Price: <b>{price_str}</b></div>"
            f"{error_html}"
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>Triggered by {triggered}</div>"
        )
    if etype == "tool_call":
        args = entry.get("args") or {}
        result = entry.get("result") or {}
        parts = []
        if args:
            parts.append(f"<div style='margin-bottom:2px'>{_kv_row_html(args)}</div>")
        if "error" in result:
            parts.append(
                f"<div style='color:{PALETTE['down']}'>⚠ {html.escape(str(result['error']))}</div>"
            )
        elif set(result.keys()) <= {"note"} and result.get("note"):
            parts.append(
                f"<div style='color:{PALETTE['muted']};font-style:italic'>{html.escape(str(result['note']))}</div>"
            )
        elif result:
            parts.append(f"<div>{_kv_row_html(result)}</div>")
        return "".join(parts)
    if etype == "regime_select":
        label = html.escape(str(entry.get("label", entry.get("strategy", ""))))
        regime = html.escape(str(entry.get("regime", "unknown")))
        reasoning = html.escape(entry.get("reasoning", ""))
        return (
            f"<div>Activated <b>{label}</b> · Regime: <b>{regime}</b></div>"
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
        )
    if etype == "stand_down":
        label = html.escape(_personality_label(str(entry.get("personality", ""))))
        quiet = entry.get("expected_quiet_minutes")
        quiet_str = f" · ~{quiet:g} min quiet expected" if isinstance(quiet, (int, float)) else ""
        reasoning = html.escape(entry.get("reasoning", ""))
        return (
            f"<div><b>{label}</b> relinquished control{quiet_str}</div>"
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{reasoning}</div>"
        )
    if etype in ("analysis", "error", "status", "cycle_start", "news_alert"):
        return f"<div>{html.escape(entry.get('text', ''))}</div>"
    return ""


def _agent_log_html(log: list[dict]) -> str:
    if not log:
        return f"<p style='color:{PALETTE['muted']};padding:12px'>No agent activity yet.</p>"
    cards = []
    for entry in reversed(log):
        icon, color, label = _agent_entry_style(entry)
        try:
            ts_fmt = pd.to_datetime(entry.get("ts", "")).strftime("%H:%M:%S")
        except Exception:
            ts_fmt = str(entry.get("ts", ""))
        cards.append(
            f"""
        <div style="background:{PALETTE['panel']}; border-radius:8px; padding:10px 14px;
                    border-left:3px solid {color}; border-top:1px solid {PALETTE['grid']};
                    border-right:1px solid {PALETTE['grid']}; border-bottom:1px solid {PALETTE['grid']};
                    margin-bottom:8px; font-size:12px; color:{PALETTE['text']};">
          <div style="display:flex; justify-content:space-between; color:{PALETTE['muted']}; font-size:11px; margin-bottom:4px;">
            <span>{icon} <b style="color:{color}">{html.escape(label)}</b></span>
            <span>{ts_fmt}</span>
          </div>
          {_agent_entry_body(entry)}
        </div>"""
        )
    return (
        "<div style='font-family:Inter,sans-serif; max-height:420px; overflow-y:auto;'>"
        f"{''.join(cards)}</div>"
    )


def _current_tactics_html(state: AppState) -> str:
    """The 'current tactics' card for the Agent tab: every armed conditional
    action (across all symbols) with each plan's reasoning and arming time, or
    an explicit nothing-armed note while the agent runs (idle agent renders
    nothing)."""
    armed_blocks: list[str] = []
    for sym_state in state.iter_symbol_states():
        tactics = sym_state.tactics
        armed = tactics_summaries(tactics)
        if not armed:
            continue
        try:
            since = f" · armed {pd.to_datetime(tactics.ts).strftime('%H:%M:%S')}" if tactics.ts else ""
        except Exception:
            since = ""
        items = "".join(f"<div>🎯 <b>{html.escape(t)}</b></div>" for t in armed)
        reasoning = (
            f"<div style='margin-top:4px;color:{PALETTE['muted']}'>{html.escape(tactics.reasoning)}</div>"
            if tactics.reasoning
            else ""
        )
        armed_blocks.append(
            f"<div style='margin-bottom:6px'>"
            f"<div style='color:{PALETTE['orange']};font-size:11px;margin-bottom:4px'>"
            f"{html.escape(sym_state.symbol)}{since}</div>"
            f"{items}{reasoning}</div>"
        )

    if not armed_blocks:
        if not state.agent_running:
            return ""
        return (
            f"<div style='background:{PALETTE['panel']};border:1px dashed {PALETTE['grid']};"
            "border-radius:8px;padding:8px 14px;margin-bottom:8px;font-size:12px;"
            f"color:{PALETTE['muted']}'>🎯 No tactics armed — the agent has not set "
            "buy/sell conditions on any symbol; it will act only when woken "
            "(alert, news, or timer).</div>"
        )
    return (
        f"<div style='background:{PALETTE['panel']};border:1px solid {PALETTE['orange']};"
        "border-radius:8px;padding:8px 14px;margin-bottom:8px;font-size:12px;"
        f"color:{PALETTE['text']}'>"
        f"<div style='color:{PALETTE['orange']};font-size:11px;margin-bottom:4px'>"
        f"ARMED TACTICS — execute automatically, then wake the agent</div>"
        f"{''.join(armed_blocks)}</div>"
    )


@st.fragment(run_every=AGENT_LOG_POLL_SEC)
def _agent_identity_panel() -> None:
    """Avatar card for the personality currently in charge. Under Automatic the
    face shown is the strategy Automatic activated, not Automatic itself; while
    it is still classifying the regime (or idle) the orchestrator's own avatar
    shows. Polled so the card follows Automatic's strategy switches live."""
    state = _get_state()
    selected = state.llm_personality
    display_key = selected
    note = ""
    if selected == AUTOMATIC_KEY and state.agent_running:
        active = state.automatic_active_strategy
        if active:
            display_key = active
            regime = f" — {state.automatic_regime} regime" if state.automatic_regime else ""
            note = f"🤖 picked by Automatic{regime}"
        else:
            note = "🤖 Automatic is assessing the market regime…"
    avatar = _avatar_data_uri(display_key)
    img = (
        f"<img src='{avatar}' alt='' style='width:56px;height:56px;border-radius:50%;flex:none'/>"
        if avatar
        else ""
    )
    note_html = (
        f"<div style='color:{PALETTE['muted']};font-size:0.85rem'>{html.escape(note)}</div>"
        if note
        else ""
    )
    st.html(
        f"<div style='display:flex;align-items:center;gap:14px;background:{PALETTE['panel']};"
        f"border:1px solid {PALETTE['grid']};border-radius:12px;padding:10px 16px;margin:4px 0'>"
        f"{img}"
        f"<div>"
        f"<div style='color:{PALETTE['text']};font-weight:600;font-size:1.05rem'>"
        f"{html.escape(_personality_label(display_key))}</div>"
        f"{note_html}"
        f"</div></div>"
    )


@st.fragment(run_every=AGENT_LOG_POLL_SEC)
def _agent_log_panel() -> None:
    state = _get_state()
    tracker = state.decision_tracker
    if tracker:
        snap = tracker.snapshot()
        positions = {s: q for s, q in snap["positions"].items() if q}
        c1, c2, c3 = st.columns(3)
        c1.metric("Paper cash", f"${snap['cash']:,.2f}")
        c2.metric(
            "Positions",
            " · ".join(f"{s} {q:.2f} sh" for s, q in positions.items()) if positions else "flat",
        )
        c3.metric("Decisions", len(snap["decisions"]))
    tactics_html = _current_tactics_html(state)
    if tactics_html:
        st.html(tactics_html)
    with state.lock:
        log = list(state.agent_log)
    st.html(_agent_log_html(log[-50:]))


def _record_live_equity_point(state: AppState) -> None:
    """Append a snapshot of the agent's current total value (all positions marked
    to their live prices), so the chart keeps advancing every poll instead of
    waiting on a full bar to close (bars can lag a minute or more behind)."""
    value = state.mark_to_market()
    if value is None:
        return
    tracker = state.decision_tracker
    snap = tracker.snapshot() if tracker else {"cash": 0.0, "positions": {}}
    with state.lock:
        state.agent_equity_history.append(
            {
                "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                "price": None,
                "cash": snap["cash"],
                "position": sum(snap["positions"].values()),
                "value": value,
            }
        )
        if len(state.agent_equity_history) > AGENT_EQUITY_HISTORY_MAXLEN:
            state.agent_equity_history = state.agent_equity_history[-AGENT_EQUITY_HISTORY_MAXLEN:]


def _merge_live_history(state: AppState, points: list[dict], agent_start: datetime) -> list[dict]:
    with state.lock:
        history = [h for h in state.agent_equity_history if pd.Timestamp(h["ts"]) > pd.Timestamp(agent_start)]
    return sorted(points + history, key=lambda p: p["ts"])


def _bars_by_symbol(state: AppState) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for sym_state in state.iter_symbol_states():
        with sym_state.lock:
            result[sym_state.symbol] = list(sym_state.bars)
    return result


@st.fragment(run_every=AGENT_PERFORMANCE_POLL_SEC)
def _agent_performance_panel(symbols: list[str]) -> None:
    state = _get_state()
    tracker = state.decision_tracker
    if not tracker:
        st.plotly_chart(empty_chart("Start the agent to track performance"), width='stretch')
        return

    snap = tracker.snapshot()
    decisions = [asdict(d) for d in snap["decisions"]]
    bars_by_symbol = _bars_by_symbol(state)

    agent_start = state.agent_start_time or SESSION_START
    _record_live_equity_point(state)
    points = compute_equity_curve(bars_by_symbol, decisions, state.starting_budget, agent_start)
    points = _merge_live_history(state, points, agent_start)
    markers = decision_markers(decisions, agent_start, points)
    stats = summarize(points, decisions, state.starting_budget)

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio value", f"${stats['current_value']:,.2f}", f"{stats['return_pct']:+.2f}%")
    c2.metric("Fees paid", f"${stats['total_fees']:,.2f}")
    c3.metric("Starting budget", f"${stats['starting_cash']:,.2f}")

    label = ", ".join(symbols or state.symbols)
    fig = build_performance_chart(points, markers, label)
    st.plotly_chart(fig, width='stretch')


def _build_agent_report_html(state: AppState, symbols: list[str]) -> str:
    syms = symbols or list(state.symbols)

    with state.lock:
        agent_log = list(state.agent_log)

    tracker = state.decision_tracker
    decisions = [asdict(d) for d in tracker.snapshot()["decisions"]] if tracker else []

    live_figs: list[tuple[str, object]] = []
    for sym in syms:
        sym_state = state.sym(sym)
        if sym_state is None:
            continue
        with sym_state.lock:
            bars = list(sym_state.bars)
            trades = list(sym_state.trades)
        if not bars:
            continue
        live_figs.append(
            (
                sym,
                build_chart(
                    bars,
                    sym_state.news,
                    trades,
                    sym,
                    SESSION_START,
                    ma_periods=state.ma_periods,
                    show_fib=state.show_fib,
                    show_7d_avg=state.show_7d_avg,
                    show_28d_avg=state.show_28d_avg,
                    show_1y_avg=state.show_1y_avg,
                    mixture_distribution=state.mixture_distribution,
                    mixture_max_components=state.mixture_max_components,
                    predicted_profile=(
                        predicted_open_profile(sym_state, bars)
                        if state.show_predicted_profile
                        else None
                    ),
                    mixture_fit_target=state.mixture_fit_target,
                    daily_bars=sym_state.daily_bars,
                    vwap_style=state.vwap_style,
                    show_candle_body=state.show_candle_body,
                    show_percentile_body=state.show_percentile_body,
                    show_whiskers=state.show_whiskers,
                    decisions=tracker.trade_markers(symbol=sym) if tracker else None,
                    news_impacts=sym_state.news_impacts,
                    fill_gaps=state.fill_gaps,
                ),
            )
        )

    historical_period_label = st.session_state.get("hist_period")
    historical_figs: list[tuple[str, object]] = []
    if historical_period_label:
        days = HISTORICAL_PERIODS[historical_period_label]
        target_syms = st.session_state.get("hist_target_syms") or []
        try:
            spy_close = fetch_close_series(SPY_SYMBOL, days)
            vix_close = fetch_close_series(VIX_SYMBOL, days)
        except Exception:
            spy_close, vix_close = None, None
        for sym in syms:
            try:
                ticker_close = fetch_close_series(sym, days)
                dividends = fetch_dividends(sym, days)
                earnings = fetch_earnings_dates(sym, days)
                price_targets = fetch_price_target_history(sym, days) if sym in target_syms else None
                historical_figs.append(
                    (
                        sym,
                        build_historical_chart(
                            ticker_close,
                            spy_close if sym != SPY_SYMBOL else None,
                            vix_close,
                            sym,
                            historical_period_label,
                            dividends,
                            earnings,
                            price_targets=price_targets,
                        ),
                    )
                )
            except Exception:
                continue

    performance_fig = None
    performance_stats = None
    agent_start = state.agent_start_time or SESSION_START
    if tracker:
        points = compute_equity_curve(_bars_by_symbol(state), decisions, state.starting_budget, agent_start)
        points = _merge_live_history(state, points, agent_start)
        markers = decision_markers(decisions, agent_start, points)
        performance_stats = summarize(points, decisions, state.starting_budget)
        performance_fig = build_performance_chart(points, markers, ", ".join(syms))

    return build_report_html(
        symbols=syms,
        feed=state.feed,
        timeframe=state.timeframe,
        session_start=agent_start,
        starting_budget=state.starting_budget,
        trade_fixed_cost=TRADE_FIXED_COST,
        llm_provider=state.llm_provider,
        llm_model=state.llm_model,
        llm_personality=_personality_label(state.llm_personality),
        agent_running=state.agent_running,
        live_figs=live_figs,
        historical_figs=historical_figs,
        historical_period_label=historical_period_label,
        performance_fig=performance_fig,
        performance_stats=performance_stats,
        decisions=decisions,
        agent_log=agent_log,
    )


def _agent_report_section(symbols: list[str]) -> None:
    state = _get_state()
    st.divider()
    st.caption("Save everything about this run — charts, starting conditions, and the full decision history — to a single HTML file.")
    if st.button("📄 Generate Report", key="agent_generate_report"):
        with st.spinner("Building report…"):
            try:
                report_html = _build_agent_report_html(state, symbols)
            except Exception as exc:
                st.error(f"Failed to build report: {exc}")
            else:
                label = "_".join(symbols or state.symbols) or "agent"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["agent_report_html"] = report_html
                st.session_state["agent_report_name"] = f"{label}_agent_report_{ts}.html"

    report_html = st.session_state.get("agent_report_html")
    if report_html:
        st.download_button(
            "💾 Save Report (.html)",
            data=report_html,
            file_name=st.session_state.get("agent_report_name", "agent_report.html"),
            mime="text/html",
            key="agent_report_download",
        )


def _agent_panel(
    symbols: list[str], alpaca_key: str = "", alpaca_secret: str = "", feed: str = "iex"
) -> None:
    state = _get_state()
    st.caption(
        "Runs an LLM research agent that reads already-fetched data for every streamed "
        "ticker and makes paper buy/sell/alert calls on a fixed interval, allocating one "
        "shared cash balance across the whole basket. Instead of trading only at the "
        "current price, the agent prefers to arm tactics -- standing conditional orders "
        "like 'buy 10 sh AAPL if price below X' or 'sell 20% of TSLA shares if price above Y' "
        "with multiple AND-ed conditions (price, volume, VIX, momentum, ...) -- that a "
        "background executor fills the instant they trigger, waking the agent to reevaluate. "
        "When it doesn't want to trade, the agent sets condition alerts on any ticker's "
        "continuously-updated values to wake up early the moment one is crossed, and it "
        "always wakes up early when fresh news breaks for any of its tickers. No real "
        "orders are ever placed. "
        f"Each filled buy/sell costs a fixed ${TRADE_FIXED_COST:.2f}."
    )
    with st.expander("LLM", expanded=True):
        # Automatic first: it's the regime-adaptive orchestrator that picks and
        # switches between the individual strategies on its own.
        personality_keys = [AUTOMATIC_KEY, *selectable_personalities()]
        personality = st.selectbox(
            "Personality",
            personality_keys,
            index=personality_keys.index(state.llm_personality)
            if state.llm_personality in personality_keys
            else personality_keys.index(DEFAULT_PERSONALITY),
            format_func=_personality_label,
            key="agent_llm_personality",
        )
        state.llm_personality = personality
        if personality == AUTOMATIC_KEY:
            st.caption(
                "🤖 Automatic detects the market regime and activates the best-fitting "
                "strategy. That strategy trades until it sees no opportunities in the near "
                "term and stands down, waking Automatic to re-assess and switch. Before "
                "the session starts it activates the Premarket Analyst instead."
            )
        elif personality == PREMARKET_PERSONALITY:
            st.caption(
                "🌅 The Premarket Analyst doesn't analyze on start: it holds until "
                "~2 minutes before the opening bell, then runs one pre-market read and "
                "arms opening tactics — how much of each ticker to buy/sell and at what "
                "price for the later trades to be profitable. Once a tactic executes "
                "(simulated at the opening prints), the analyst retires and the agent "
                "disables itself."
            )
        provider = st.selectbox(
            "Provider", PROVIDERS, index=PROVIDERS.index(state.llm_provider), key="agent_llm_provider"
        )
        default_model = DEFAULT_AGENT_MODELS[provider]
        model_options = models_for(provider, default=default_model)
        current_model = state.llm_model if state.llm_model in model_options else default_model
        model = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(current_model),
            # Key is provider-scoped so switching providers rebuilds the widget
            # instead of carrying a stale value that isn't in the new options.
            key=f"agent_llm_model_{provider}",
            help=f"Default: {default_model}",
        )
        state.llm_provider = provider
        state.llm_model = model
        env_var = ENV_KEYS[provider]
        if not os.getenv(env_var):
            st.caption(f"⚠️ {env_var} is not set.")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    starting_budget = c1.number_input(
        "Starting budget ($)",
        min_value=0.0,
        value=PAPER_STARTING_CASH,
        step=100.0,
        key="agent_starting_budget",
    )
    start_clicked = c2.button("▶ Start Agent", type="primary", width='stretch', key="agent_start")
    stop_clicked = c3.button("⏹ Stop Agent", width='stretch', key="agent_stop")

    env_var = ENV_KEYS[provider]
    llm_key = os.getenv(env_var, "")

    if start_clicked:
        syms = list(symbols or state.symbols)
        stream_ready = False
        if not syms:
            st.error("Enter at least one symbol in the sidebar first.")
        elif not llm_key:
            st.error(f"{env_var} is not set; the agent needs an LLM key to reason about decisions.")
        else:
            # The live stream feeds every tool the agent reads. If it isn't
            # running for these symbols yet, start it here rather than sending
            # the user back to the sidebar first.
            stream_ready = bool(state.api_key) and all(state.sym(s) is not None for s in syms)
            if not stream_ready:
                key = alpaca_key.strip() or os.getenv("ALPACA_API_KEY", "")
                secret = alpaca_secret.strip() or os.getenv("ALPACA_SECRET", "")
                if not key or not secret:
                    st.error(
                        "Alpaca API key and secret are required to start the live stream "
                        "(sidebar Connection expander, or the ALPACA_API_KEY / "
                        "ALPACA_SECRET environment variables)."
                    )
                else:
                    timeframe = st.session_state.get("live_timeframe", TIMEFRAMES[0])
                    stream_ready = _start_live_session(
                        state, syms, key, secret, feed, timeframe
                    )
        if stream_ready:
            if not market_hours.is_market_open():
                open_et = market_hours.next_market_open().astimezone(market_hours.MARKET_TZ)
                st.info(
                    f"The trading session hasn't started yet (next open: "
                    f"{open_et.strftime('%a %Y-%m-%d %H:%M')} ET). The agent is told the "
                    "market is closed and adapts: the Premarket Analyst prepares opening "
                    "tactics, other strategies study structure and arm plans for the "
                    "open instead of trading the stale tape."
                )
            state.starting_budget = starting_budget
            state.decision_tracker = DecisionTracker(starting_cash=starting_budget, trade_cost=TRADE_FIXED_COST)
            state.agent_log = []
            state.agent_start_time = datetime.now(tz=timezone.utc)
            state.agent_equity_history = []
            if personality == AUTOMATIC_KEY:
                launch_automatic(
                    state,
                    state.decision_tracker,
                    syms,
                    llm_key,
                    provider=provider,
                    model=model or None,
                    cycle_sec=AGENT_CYCLE_SEC,
                )
            else:
                launch_agent(
                    state,
                    state.decision_tracker,
                    syms,
                    llm_key,
                    provider=provider,
                    model=model or None,
                    cycle_sec=AGENT_CYCLE_SEC,
                    personality=personality,
                )

    if stop_clicked:
        stop_agent(state)

    status = "🟢 running" if state.agent_running else "⚪ idle"
    watching = (
        f" — watching {', '.join(state.symbols)}"
        if state.agent_running and state.symbols
        else ""
    )
    st.caption(f"Status: {status}{watching}")
    _agent_identity_panel()

    _agent_performance_panel(symbols)
    _agent_log_panel()
    _agent_report_section(symbols)


_BIAS_STYLE: dict[str, dict[str, str]] = {
    "bullish":  {"color": "#26c6a2", "bg": "#0d2b24", "border": "#1a4a3d", "icon": "▲"},
    "bearish":  {"color": "#ef5350", "bg": "#2b0d0d", "border": "#4a1a1a", "icon": "▼"},
    "neutral":  {"color": "#888888", "bg": "#1e1e2e", "border": "#2a2d3a", "icon": "→"},
}

_CONF_COLOR: dict[str, str] = {"high": "#26c6a2", "medium": "#fb923c", "low": "#888888"}

_IMPACT_ICON: dict[str, str] = {"positive": "↑", "negative": "↓", "neutral": "→"}


def _premarket_briefing_html(briefing: PremarketBriefing, symbol: str) -> str:
    bias = briefing.overall_bias
    b = _BIAS_STYLE.get(bias, _BIAS_STYLE["neutral"])
    conf_color = _CONF_COLOR.get(briefing.confidence, "#888")

    # Header
    header = (
        f'<div style="background:{b["bg"]};border:1px solid {b["border"]};border-radius:10px;'
        f'padding:14px 18px;margin-bottom:12px;font-family:Inter,sans-serif;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
        f'<span style="font-size:22px;font-weight:800;color:{b["color"]};letter-spacing:-0.5px;">'
        f'{b["icon"]} {bias.upper()}</span>'
        f'<span style="font-size:11px;font-weight:600;color:{conf_color};'
        f'border:1px solid {conf_color};border-radius:10px;padding:2px 8px;">'
        f'{briefing.confidence.upper()} CONFIDENCE</span>'
        f'<span style="font-size:11px;color:{PALETTE["muted"]};margin-left:auto;">{symbol}</span>'
        f'</div>'
        f'<p style="margin:0;color:{PALETTE["text"]};font-size:13px;line-height:1.6;">'
        f'{html.escape(briefing.summary)}</p>'
        f'</div>'
    )

    # Macro
    macro = (
        f'<div style="background:{PALETTE["panel"]};border:1px solid {PALETTE["grid"]};'
        f'border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:12px;'
        f'color:{PALETTE["muted"]};font-family:Inter,sans-serif;">'
        f'🌍 {html.escape(briefing.macro_context)}</div>'
    )

    # Catalysts
    catalyst_cards = []
    for c in briefing.catalysts:
        imp = c.impact
        imp_color = _IMPACT_STYLE.get(imp, _IMPACT_STYLE["unknown"])
        icon = _IMPACT_ICON.get(imp, "→")
        catalyst_cards.append(
            f'<div style="background:{imp_color["bg"]};border:1px solid {imp_color["border"]};'
            f'border-radius:8px;padding:10px 12px;font-family:Inter,sans-serif;">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
            f'<span style="color:{imp_color["text"]};font-weight:700;font-size:12px;">{icon}</span>'
            f'<span style="color:{PALETTE["text"]};font-weight:600;font-size:12px;">'
            f'{html.escape(c.headline)}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:{PALETTE["muted"]};line-height:1.4;">'
            f'{html.escape(c.relevance)}</div>'
            f'</div>'
        )
    catalysts_section = (
        f'<div style="margin-bottom:12px;">'
        f'<div style="font-size:12px;font-weight:700;color:{PALETTE["muted"]};'
        f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Key Catalysts</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px;">'
        f"{''.join(catalyst_cards)}"
        f'</div></div>'
    ) if catalyst_cards else ""

    # Technical levels
    level_rows = ""
    for lvl in briefing.technical_levels:
        role_color = "#26c6a2" if "support" in lvl.role.lower() else "#ef5350" if "resist" in lvl.role.lower() else PALETTE["accent"]
        level_rows += (
            f'<tr>'
            f'<td style="padding:5px 10px;font-weight:700;color:{role_color};">${lvl.level:.2f}</td>'
            f'<td style="padding:5px 10px;color:{PALETTE["muted"]};text-transform:capitalize;">{html.escape(lvl.role)}</td>'
            f'<td style="padding:5px 10px;color:{PALETTE["text"]};">{html.escape(lvl.note)}</td>'
            f'</tr>'
        )
    levels_section = (
        f'<div style="margin-bottom:12px;">'
        f'<div style="font-size:12px;font-weight:700;color:{PALETTE["muted"]};'
        f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Technical Levels</div>'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,monospace;font-size:12px;'
        f'background:{PALETTE["panel"]};border-radius:8px;overflow:hidden;">'
        f'{level_rows}</table></div>'
    ) if level_rows else ""

    # Risk factors + watch list side by side
    def _bullet_list(items: list[str], title: str, icon: str = "⚠️") -> str:
        if not items:
            return ""
        lis = "".join(
            f'<li style="margin-bottom:4px;line-height:1.4;">{html.escape(r)}</li>'
            for r in items
        )
        return (
            f'<div style="flex:1;background:{PALETTE["panel"]};border:1px solid {PALETTE["grid"]};'
            f'border-radius:8px;padding:10px 14px;font-family:Inter,sans-serif;">'
            f'<div style="font-size:12px;font-weight:700;color:{PALETTE["muted"]};'
            f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">{icon} {title}</div>'
            f'<ul style="margin:0;padding-left:16px;font-size:12px;color:{PALETTE["text"]};">{lis}</ul>'
            f'</div>'
        )

    risks = _bullet_list(briefing.risk_factors, "Risk Factors", "⚠️")
    watch = _bullet_list(briefing.key_levels_to_watch, "Watch During Session", "👁️")
    bottom_row = (
        f'<div style="display:flex;gap:10px;margin-bottom:12px;">{risks}{watch}</div>'
        if (risks or watch) else ""
    )

    return (
        f'<div style="font-family:Inter,sans-serif;padding:4px 0 16px;">'
        f'<h3 style="color:{PALETTE["text"]};font-size:14px;margin:0 0 10px 0">'
        f'🌅 Pre-Market Briefing · <b style="color:{PALETTE["accent"]}">{symbol}</b>'
        f'</h3>'
        f'{header}{macro}{catalysts_section}{levels_section}{bottom_row}'
        f'</div>'
    )


def _premarket_panel(symbols: list[str]) -> None:
    state = _get_state()

    st.caption(
        "Synthesizes recent news, historical price action, macro indicators, and fundamentals "
        "into a structured morning briefing per symbol. Works any time — no live stream needed."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        provider = st.selectbox(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(state.news_llm_provider),
            key="premarket_provider",
            help=f"Model: {', '.join(f'{p}={m}' for p, m in DEFAULT_PREMARKET_MODELS.items())}",
        )
        state.news_llm_provider = provider
        env_var = ENV_KEYS[provider]
        llm_key = os.getenv(env_var, "")
        if not llm_key:
            st.caption(f"⚠️ {env_var} is not set.")
        default_model = DEFAULT_PREMARKET_MODELS.get(provider, DEFAULT_NEWS_MODELS[provider])
        premarket_models = models_for(provider, default=default_model)
        model_override = st.selectbox(
            "Model",
            premarket_models,
            index=0,
            key=f"premarket_model_{provider}",
            help=f"Default: {default_model}",
        )
    with c2:
        if not symbols:
            st.info("Enter symbols in the sidebar first.")
        else:
            generate_clicked = st.button(
                f"🌅 Generate Pre-Market Analysis ({', '.join(symbols)})",
                key="premarket_generate",
                type="primary",
            )
            if generate_clicked:
                if not llm_key:
                    st.error(f"{env_var} is not set.")
                else:
                    briefings: dict[str, PremarketBriefing] = {}
                    for sym in symbols:
                        with st.spinner(f"Generating pre-market briefing for {sym}…"):
                            try:
                                briefing = generate_premarket_analysis(
                                    symbol=sym,
                                    provider=provider,
                                    api_key=llm_key,
                                    alpaca_key=state.api_key or os.getenv("ALPACA_API_KEY", ""),
                                    alpaca_secret=state.api_secret or os.getenv("ALPACA_SECRET", ""),
                                    worldnews_key=os.getenv("WORLD_NEWS_API_KEY", ""),
                                    model=model_override or None,
                                )
                            except Exception as exc:
                                st.error(f"Pre-market analysis failed for {sym}: {exc}")
                            else:
                                if briefing is not None:
                                    briefings[sym] = briefing
                    if briefings:
                        st.session_state["premarket_briefings"] = briefings

    briefings = st.session_state.get("premarket_briefings") or {}
    for sym, briefing in briefings.items():
        st.html(_premarket_briefing_html(briefing, sym))


def _start_live_session(
    state: AppState, syms: list[str], key: str, secret: str, feed: str, timeframe: str
) -> bool:
    """Load history for every symbol and launch the bars + news streams.

    Shared by the sidebar ▶ Start and the agent panel's ▶ Start Agent (which
    starts the stream itself when it isn't running yet). Returns True when the
    streams were launched, False when loading any symbol failed."""
    state.set_symbols(syms)
    state.feed = feed
    state.timeframe = timeframe
    state.api_key = key
    state.api_secret = secret
    loaded: list[str] = []
    for sym in syms:
        sym_state = state.sym(sym)
        with st.spinner(f"Loading history for {sym}…"):
            try:
                historical_bars = fetch_bars(sym, timeframe, MAX_BARS, key, secret, feed)
                log_fetch(
                    "bars (initial load)", "Alpaca REST", symbol=sym,
                    detail=f"{len(historical_bars)} {timeframe} bars",
                )
                historical_trades = fetch_trades(sym, key, secret, feed)
                log_fetch(
                    "trades (initial load)", "Alpaca REST", symbol=sym,
                    detail=f"{len(historical_trades)} trades",
                )
                news = fetch_news_with_fallback(
                    sym, key, secret, os.getenv("WORLD_NEWS_API_KEY", "")
                )
                daily_bars = fetch_daily_bars(sym, key, secret, feed)
                log_fetch(
                    "daily bars (initial load)", "Alpaca REST", symbol=sym,
                    detail=f"{len(daily_bars)} daily bars",
                )
            except Exception as exc:
                log_fetch_failure(
                    "initial data load", [("Alpaca REST", exc)], symbol=sym,
                    consequence="start aborted",
                )
                st.error(f"Failed to load data for {sym}: {exc}")
                state.status = f"Failed ({sym}): {exc}"
                return False
            sym_state.daily_bars = daily_bars
            with sym_state.lock:
                sym_state.bars.clear()
                sym_state.bars.extend(historical_bars)
                sym_state.trades.clear()
                sym_state.trades.extend(historical_trades)
            sym_state.news = news
            sym_state.news_impacts = {}
            loaded.append(sym)

    news_llm_provider = state.news_llm_provider
    llm_key = os.getenv(ENV_KEYS[news_llm_provider], "")
    if llm_key:
        for sym in loaded:
            sym_state = state.sym(sym)
            if not sym_state.news:
                continue
            try:
                sym_state.news_impacts = score_news_impacts(
                    sym, sym_state.news, news_llm_provider, llm_key
                )
            except Exception as exc:
                state.status = f"News impact scoring failed for {sym}: {exc}"
    state.status = "Connecting WebSocket…"
    launch_stream(syms, key, secret, feed, state, timeframe)
    launch_stream_news(
        syms, key, secret, state, worldnews_key=os.getenv("WORLD_NEWS_API_KEY", "")
    )
    return True


def build_ui() -> None:
    st.set_page_config(
        page_title="Agent Stonks",
        page_icon="📈",
        layout="wide",
    )

    with st.sidebar:
        st.header("Controls")
        c1, c2 = st.columns(2)
        start_clicked = c1.button("▶ Start", type="primary", width='stretch')
        stop_clicked = c2.button("⏹ Stop", width='stretch')
        symbols_input = st.text_input(
            "Symbols",
            value="AAPL",
            placeholder="AAPL, TSLA, MSFT…",
            help="One or more tickers, comma- or space-separated. All live plots, "
            "analyses, and the trading agent cover every listed symbol.",
        )
        with st.expander("Connection"):
            feed = st.selectbox("Feed", FEEDS, index=0)
            api_key = st.text_input(
                "Alpaca API Key",
                type="password",
                placeholder="From env ALPACA_API_KEY if blank",
            )
            api_secret = st.text_input(
                "Alpaca Secret",
                type="password",
                placeholder="From env ALPACA_SECRET if blank",
            )
    state = _get_state()
    symbols = _effective_symbols(state, symbols_input)

    tab_live, tab_news, tab_premarket, tab_historical, tab_analysis, tab_smart_money, tab_walls, tab_agent = st.tabs(
        ["📡 Live", "📰 News", "🌅 Pre-Market", "🗂️ Historical", "🔬 Technical Analysis", "🏦 Smart Money", "🧱 Put/Call Walls", "🤖 Agent"]
    )

    with tab_live:
        st.caption(
            "⚠️ Free Alpaca accounts: IEX feed available during US market hours (9:30–16:00 ET). "
        )
        timeframe = st.session_state.get("live_timeframe", TIMEFRAMES[0])

        timeframe_changed = (
            state.symbols
            and state.api_key
            and timeframe != state.timeframe
            and not start_clicked
            and not stop_clicked
        )
        if timeframe_changed:
            with st.spinner(f"Reloading {', '.join(state.symbols)} at {timeframe}…"):
                reloaded = True
                for sym_state in state.iter_symbol_states():
                    sym = sym_state.symbol
                    try:
                        historical_bars = fetch_bars(
                            sym, timeframe, MAX_BARS, state.api_key, state.api_secret, state.feed
                        )
                    except Exception as exc:
                        log_fetch_failure(
                            "bars (timeframe reload)", [("Alpaca REST", exc)], symbol=sym,
                            consequence=f"staying on {state.timeframe}",
                        )
                        st.error(f"Failed to reload bars for {sym}: {exc}")
                        reloaded = False
                        break
                    log_fetch(
                        "bars (timeframe reload)", "Alpaca REST", symbol=sym,
                        detail=f"{len(historical_bars)} {timeframe} bars",
                    )
                    with sym_state.lock:
                        sym_state.bars.clear()
                        sym_state.bars.extend(historical_bars)
                if reloaded:
                    state.timeframe = timeframe
                    launch_stream(
                        list(state.symbols), state.api_key, state.api_secret,
                        state.feed, state, timeframe,
                    )

        if start_clicked:
            syms = _parse_symbols(symbols_input)
            key = api_key.strip() or os.getenv("ALPACA_API_KEY", "")
            secret = api_secret.strip() or os.getenv("ALPACA_SECRET", "")

            if not syms:
                st.error("Please enter at least one symbol.")
            elif not key or not secret:
                st.error("API key and secret are required.")
            else:
                _start_live_session(state, syms, key, secret, feed, timeframe)

        if stop_clicked:
            if state.bars_fallback_stop_event:
                state.bars_fallback_stop_event.set()
            if state.news_fallback_stop_event:
                state.news_fallback_stop_event.set()
            if state.ws:
                try:
                    state.ws.close()
                except Exception:
                    pass
            if state.ws_news:
                try:
                    state.ws_news.close()
                except Exception:
                    pass
            state.status = "Stopped"
            state.news_status = "Stopped"

        _live_panel()

    with tab_news:
        _news_panel(symbols)

    with tab_premarket:
        _premarket_panel(symbols)

    with tab_historical:
        _historical_panel(symbols)

    with tab_analysis:
        _technical_analysis_panel(symbols)

    with tab_smart_money:
        _smart_money_panel(symbols)

    with tab_walls:
        _options_walls_panel(symbols)

    with tab_agent:
        _agent_panel(symbols, alpaca_key=api_key, alpaca_secret=api_secret, feed=feed)
