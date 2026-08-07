"""SimLab Streamlit UI: agents / datasets / simulate / summary / results.

Run with ``streamlit run sim_main.py``. Kept separate from the live dashboard
(``main.py``) -- this app never opens a stream or touches the live tape; it
only reads the local dataset store and replays agents against it.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta, timezone
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent_stonks import apple_models, clock, persistence_model
from agent_stonks import observability as obs
from agent_stonks.agent import (
    AGENT_PERSONALITIES,
    PERSONALITY_TOOLS,
    _dispatch_tool,
    selectable_personalities,
)
from agent_stonks.apple_trader import (
    APPLE_TRADER_KEY,
    ENTRY_ANTICIPATE,
    ENTRY_MODE_LABEL,
    ENTRY_MODE_PROB_LABEL,
    ENTRY_MODE_SUMMARY,
    ENTRY_MODES,
    RULE_PROVIDER,
    AppleTraderConfig,
)
from agent_stonks.claude_rule_trader import (
    TRADER_BY_CLAUDE_KEY,
    TraderByClaudeConfig,
)
from agent_stonks.config import PALETTE
from agent_stonks.openai_rule_trader import (
    TRADER_BY_CHATGPT_KEY,
    TraderByChatGPTConfig,
)
from agent_stonks.llm import DEFAULT_AGENT_MODELS, ENV_KEYS, PROVIDERS, models_for
from agent_stonks.market_hours import MARKET_TZ

from . import data as sim_data
from . import experiments as sim_experiments
from . import prompts as sim_prompts
from . import results as sim_results
from .engine import SimulationConfig, SimulationEngine
from .market import SimMarket
from .patches import simulation_context
from .rule_agents import RULE_AGENTS, rule_agent

AVATAR_DIR = Path(__file__).resolve().parent.parent / "data" / "avatars"

# submit_decision / set_tactics mutate the ledger and need a full cycle around
# them -- the hand-tester exposes only the read/analysis tools.
_UNTESTABLE_TOOLS = {"submit_decision", "set_tactics", "stand_down"}

def _testable_agents() -> list[str]:
    """Every agent SimLab can replay, in picker order: the LLM personalities
    first, the rule-based ones last."""
    return [*selectable_personalities(), *RULE_AGENTS]


def _agent_label(key: "str | None") -> str:
    """Rule agents have no entry in AGENT_PERSONALITIES, so their label comes
    from the registry instead."""
    if key in RULE_AGENTS:
        return RULE_AGENTS[key].label
    return AGENT_PERSONALITIES.get(key or "", {}).get("label", key or "?")


def _agent_avatar(key: str) -> Path:
    if key in RULE_AGENTS:
        return AVATAR_DIR / RULE_AGENTS[key].avatar
    return AVATAR_DIR / AGENT_PERSONALITIES.get(key, {}).get("avatar", "")


def _env_key(provider: str) -> str:
    return os.getenv(ENV_KEYS.get(provider, ""), "")


@st.cache_data(show_spinner=False)
def _load_runs(signature: tuple) -> list[dict]:
    # `signature` is unused on purpose: it is the cache key, so the store is
    # re-read exactly when a run file is added, changed, or deleted. It must
    # not be underscore-prefixed -- Streamlit excludes those from the key.
    return sim_results.list_runs()


def _runs() -> list[dict]:
    """Stored run records, re-read only when the run store actually changes.
    Full records carry decisions and agent logs, so parsing every one of them
    on every rerun adds up."""
    return _load_runs(sim_results.store_signature())


def _chart_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=30, b=30),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"]),
        xaxis=dict(gridcolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"]),
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 1 — agents
# ---------------------------------------------------------------------------

def _render_tool_tester(personality: str) -> None:
    st.markdown("##### Try a tool by hand")
    datasets = sim_data.list_datasets()
    if not datasets:
        st.info("Download a dataset first (Datasets tab) to test tools against stored data.")
        return

    ds_names = [d.name for d in datasets]
    col_ds, col_sym, col_day = st.columns(3)
    ds = sim_data.get_dataset(col_ds.selectbox("Dataset", ds_names, key="tt_ds"))
    symbol = col_sym.selectbox("Symbol", ds.symbols, key="tt_sym")
    day = date.fromisoformat(col_day.selectbox("Day", ds.days or [ds.start], key="tt_day"))
    probe_time = st.slider(
        "Moment (ET)",
        min_value=time(4, 30),
        max_value=time(20, 0),
        value=time(10, 30),
        step=timedelta(minutes=5),
        key="tt_time",
    )
    at = datetime.combine(day, probe_time, tzinfo=MARKET_TZ).astimezone(timezone.utc)

    tools = [
        t["function"] for t in PERSONALITY_TOOLS[personality]
        if t["function"]["name"] not in _UNTESTABLE_TOOLS
    ]
    tool = st.selectbox(
        "Tool", tools, format_func=lambda t: t["name"], key="tt_tool"
    )
    with st.expander("What this tool does"):
        st.write(tool["description"])

    args: dict = {}
    props = (tool.get("parameters") or {}).get("properties", {})
    extra = {k: v for k, v in props.items() if k != "symbol"}
    if extra:
        cols = st.columns(min(3, len(extra)))
        for i, (name, spec) in enumerate(extra.items()):
            raw = cols[i % len(cols)].text_input(
                name, key=f"tt_arg_{tool['name']}_{name}",
                help=spec.get("description", ""), placeholder="default",
            )
            if raw.strip():
                try:
                    args[name] = json.loads(raw)
                except json.JSONDecodeError:
                    args[name] = raw
    args["symbol"] = symbol

    if st.button("Run tool", icon=":material/play_arrow:", type="primary", key="tt_run"):
        market = SimMarket(ds.symbols, [day], ds.feed)
        config = SimulationConfig(
            personality=personality, provider="openai", model="-", api_key="",
            symbols=ds.symbols, days=[day], feed=ds.feed,
        )
        engine = SimulationEngine(market, config)
        with simulation_context(market):
            engine.seed_until(at)
            clock.set_simulated(at)
            result = _dispatch_tool(tool["name"], args, engine.app, engine.tracker)
        st.caption(f"`{tool['name']}` at {at.astimezone(MARKET_TZ).strftime('%Y-%m-%d %H:%M ET')}")
        st.json(result)


def _render_rule_agent(personality: str) -> None:
    """A rule agent's "prompt": its rules, and whatever they depend on.

    There is nothing to edit here the way a system prompt is edited -- the
    thresholds are per-simulation settings, picked in the Simulate tab -- so
    this is a read-only description of what the loop does.
    """
    st.subheader(_agent_label(personality))
    st.caption(
        ":material/function: Rule-based — no LLM, no prompt, no tools. The same tape "
        "always produces the same trades."
    )
    if personality == TRADER_BY_CHATGPT_KEY:
        _render_chatgpt_rules()
    elif personality == TRADER_BY_CLAUDE_KEY:
        _render_claude_rules()
    else:
        _render_apple_rules()


def _render_claude_rules() -> None:
    """What TraderByClaude does, and what it is a counter-thesis to."""
    ticker = rule_agent(TRADER_BY_CLAUDE_KEY).ticker
    st.markdown(
        f"Once a minute it reads the **{ticker}** bar that just closed. Same family as "
        "TraderByChatGPT — long-only, trend-aligned, sized on distance to the stop — "
        "and a deliberate disagreement about *where in a trend you are allowed to buy*. "
        "That one buys the breakout at the highs; this one buys the pullback back into "
        "the trend:\n"
        "- **Buy** when price has dipped to **the fast mean** somewhere in the last "
        "**pullback window** bars *on below-median volume*, and this bar closes back "
        "above that mean and through the previous bar's high — inside an established "
        "uptrend (fast mean above slow mean and rising) above VWAP, and no more than "
        "**the stretch limit** ATR above the mean so it is not chasing.\n"
        "- The **quiet-pullback** test does the most work. A healthy pullback is "
        "participation drying up; a dip on heavy volume is a seller working an order, "
        "and the “support” under it is a queue.\n"
        "- **Stop** goes under the pullback low, not a fixed ATR distance below the "
        "entry — so the invalidation is structural (the low that defined the setup "
        "broke) and usually much tighter, which buys a larger position for the same "
        "risk. A setup whose stop is further than **the max risk** is skipped rather "
        "than sized down.\n"
        "- **Sell** on the stop, on a close back below the fast mean once the trade has "
        "had a few bars to work, on the time stop, or at the pre-close flatten. The "
        "stop ratchets to breakeven at **1R** and never moves back down.\n\n"
        "Its rules are set per simulation in the **Simulate** tab, and it is never "
        "scored by the LLM judge — profit, profit efficiency and the oracle ceiling are "
        "the whole verdict."
    )
    st.warning(
        ":material/warning: **What would falsify it.** The thesis is that near-support "
        "entries lose less per failure than at-resistance entries, and that this pays "
        "for the signals it skips by refusing to chase. On a tape that trends hard in "
        "one direction all session the breakout agent should win outright, because the "
        "pullback this one waits for never comes. Distinguishing those needs many "
        "sessions — a two-day dataset cannot, and neither can a two-day head-to-head."
    )


def _render_chatgpt_rules() -> None:
    """What TraderByChatGPT does. No model behind it -- nothing to load, and
    nothing whose provenance needs stating."""
    ticker = rule_agent(TRADER_BY_CHATGPT_KEY).ticker
    st.markdown(
        f"Once a minute it reads the **{ticker}** bar that just closed and asks one "
        "boolean question about it — no model, no probability, no confirmation window:\n"
        "- **Buy** when that single bar clears every condition at once: close above "
        "session VWAP, EMA9 > EMA21 > EMA50, a close through **the breakout window**'s "
        "highest high, volume at least **the relative-volume floor** times its 20-bar "
        "median, RSI inside 55–75, and ATR at or above **the minimum ATR** (tape too "
        "quiet for a breakout to travel anywhere is refused). Confluence rather than one "
        "indicator, and entries only between 09:45 and 15:15.\n"
        "- **Size** on risk, not on cash: the initial stop sits **the initial stop** ATRs "
        "below the entry, and the quantity is whatever puts **the risk budget** of equity "
        "between the fill and that stop — capped at **the position size**, since a "
        "collapsing ATR would otherwise ask for an unbounded position.\n"
        "- **Sell** on whichever comes first: the ATR stop (which ratchets up to **the "
        "trail** ATRs under the running peak once the trade is 1R ahead), a time stop "
        "after 60 bars, or the pre-close flatten.\n"
        "- A day that is 1% down stops opening new trades; whatever is already on is "
        "still managed to its stop.\n\n"
        "Its rules are set per simulation in the **Simulate** tab, and it is never scored "
        "by the LLM judge — it states no reasoning of its own to judge, so profit, profit "
        "efficiency and the oracle ceiling are the whole verdict."
    )
    st.info(
        ":material/info: Both rule agents trade the same symbol on the same tape, so a "
        "dataset run through both is a straight comparison of two hand-written strategies "
        "— one that asks a fitted model whether a momentum change will hold, one that "
        "asks nothing at all."
    )


def _render_apple_rules() -> None:
    """What Apple Trader does, plus the provenance of the bundle it needs."""
    ticker = rule_agent(APPLE_TRADER_KEY).ticker
    st.markdown(
        f"Once a minute it reads the **{ticker}** bar that just closed and "
        "tracks the momentum regime — a Schmitt trigger over a volatility-normalised "
        "momentum score, so a value hovering near the line cannot emit a burst of fake "
        "changes:\n"
        "- **Buy** on **the entry mode**'s question, given the 20 bars leading into that "
        "bar. On *Anticipate* (the default) the regime is still negative or balanced and "
        "the model forecasts that it turns positive on the next bar; on *Confirm* the "
        "change has already printed and the model rates it likely to hold. The second is "
        "the notebook's rule, and it is structurally late — the momentum score has "
        "already crossed its threshold by then, so the fill lands after the move that "
        "produced the signal.\n"
        "- **Sell** when price falls **the trailing stop** below the highest price seen "
        "since the entry. The peak only ratchets up, so the rule starts as a stop under "
        "the entry and becomes a profit lock as the move runs.\n"
        "- Nothing else closes the position but the closing bell: every feature the model "
        "uses is intraday, so the book is flattened before the close rather than carried "
        "overnight.\n\n"
        "Its rules are set per simulation in the **Simulate** tab, and it is never scored "
        "by the LLM judge — it states no reasoning of its own to judge, so profit, profit "
        "efficiency and the oracle ceiling are the whole verdict."
    )

    st.markdown("##### Models")
    st.caption(
        "The entry question can be put to either of two saved TimeToChange2 models, "
        "chosen per simulation. Everything before the question — bars, momentum, regimes, "
        "all 25 features, the 20-bar window — is identical for both, so a dataset run "
        "through each on the *Confirm* entry is a comparison of the models and nothing "
        "else. *Anticipate* asks about a bar that is not a regime change, which only a "
        "forecaster can answer."
    )
    for model in (apple_models.get(key) for key in apple_models.keys()):
        with st.expander(model.label, expanded=model.key == AppleTraderConfig().model_key):
            st.markdown(model.summary)
            if not model.anticipates:
                st.caption(
                    ":material/block: Fitted on regime-change bars only, so it runs the "
                    "*Confirm* entry and not *Anticipate*."
                )
            bundle = apple_models.load(model.key)
            if bundle is None:
                st.error(
                    f"{apple_models.unavailable_reason(model.key)} Simulations naming this "
                    "model will fail until it is available."
                )
                continue
            metrics = bundle.get("metrics") or {}
            cols = st.columns(4)
            cols[0].metric("Sequence", f"{bundle['seq_len']} bars")
            cols[1].metric("Features", len(bundle.get("feature_columns") or []))
            cols[2].metric("Held-out AUC", f"{metrics.get('roc_auc', float('nan')):.2f}")
            cols[3].metric("Own threshold", f"{persistence_model.model_threshold(bundle):g}")
            st.caption(
                f"Fitted {bundle.get('trained_at', '?')}, excluding sessions from "
                f"{bundle.get('excluded_sessions_from', '?')} onwards. "
                f"{bundle.get('notes', '')}"
            )
            st.json({"settings": bundle.get("settings"), "metrics": metrics}, expanded=False)
    st.warning(
        ":material/warning: Those AUCs cover *all* regime changes, and roughly half of "
        "them are decided by one observable boolean — the old regime had already held 15 "
        "bars. On the changes that pass it the classifier scores ~0.50 and N-BEATS ~0.67, "
        "and the interval on that 0.67 only just excludes chance. Either entry is best "
        "read as a change the model did not veto, which is why the exit does not consult "
        "it at all."
    )


def render_agents_tab() -> None:
    keys = _testable_agents()
    personality = st.session_state.get("agents_selected", keys[0])
    if personality not in keys:
        personality = keys[0]
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        with col, st.container(border=True):
            avatar = _agent_avatar(key)
            if avatar.exists():
                st.image(str(avatar), width=72)
            st.caption(_agent_label(key))
            if st.button(
                "Selected" if key == personality else "Open",
                key=f"agent_pick_{key}",
                type="primary" if key == personality else "secondary",
            ):
                st.session_state["agents_selected"] = key
                st.rerun()

    if not sim_prompts.has_prompt(personality):
        _render_rule_agent(personality)
        return

    meta = AGENT_PERSONALITIES[personality]
    st.subheader(meta["label"])
    overridden = sim_prompts.has_override(personality)
    if overridden:
        st.caption(
            ":material/edit: Using a **modified** prompt (simulations launched here use it; "
            "the live app keeps the built-in)."
        )
    else:
        st.caption(":material/lock: Using the built-in prompt.")

    prompt_text = st.text_area(
        "System prompt",
        value=sim_prompts.get_prompt(personality),
        height=420,
        key=f"prompt_editor_{personality}",
    )
    with st.container(horizontal=True):
        if st.button("Save prompt", icon=":material/save:", type="primary"):
            sim_prompts.save_override(personality, prompt_text)
            st.rerun()
        if overridden and st.button("Reset to built-in", icon=":material/restart_alt:"):
            sim_prompts.reset_override(personality)
            st.rerun()

    st.divider()
    st.markdown(f"##### Tools ({len(PERSONALITY_TOOLS[personality])})")
    st.caption(
        "The exact tool set this agent gets in a cycle. In simulation each tool reads "
        "the stored tape as of the simulated moment."
    )
    _render_tool_tester(personality)


# ---------------------------------------------------------------------------
# Tab 2 — datasets
# ---------------------------------------------------------------------------

def render_datasets_tab() -> None:
    st.caption(
        "Datasets are named bundles of symbols + a date range + a **feed**. Minute bars "
        "(04:00–20:00 ET), daily history, news, and SPY/VIX context are stored locally, "
        "deduplicated per (feed, symbol, day) — overlapping datasets never re-download a day."
    )
    st.info(
        ":material/info: **The feed is part of the data, not a download setting.** "
        "`yfinance` (the default) is the consolidated tape — every venue — free, and the "
        "same source the live volume tools read. `iex` is one venue, roughly 4% of "
        "consolidated volume on a large cap, so its bars carry different closes, far "
        "smaller volumes and occasionally an extra or missing minute; any agent whose "
        "rules are thresholds over those bars can trade a different day on each tape. "
        "The same day on two feeds is stored twice, on purpose."
    )
    st.warning(
        f":material/schedule: **yfinance reaches back {sim_data.YF_MINUTE_WINDOW_DAYS} days.** "
        "Yahoo serves 1-minute history for the last "
        f"{sim_data.YF_MINUTE_WINDOW_DAYS} days only, so a yfinance dataset cannot start "
        f"before **{date.today() - timedelta(days=sim_data.YF_MINUTE_WINDOW_DAYS)}** — "
        "earlier days download empty. Use `sip` (needs a paid Alpaca data subscription) "
        "for an older window. yfinance bars also carry no per-bar VWAP."
    )
    with st.form("dataset_form"):
        name = st.text_input("Dataset name", placeholder="e.g. nvda-earnings-week")
        symbols_raw = st.text_input("Symbols (comma-separated)", placeholder="NVDA, AAPL")
        col_start, col_end, col_feed = st.columns(3)
        start = col_start.date_input("Start", value=date.today() - timedelta(days=7))
        end = col_end.date_input("End", value=date.today() - timedelta(days=1))
        feed = col_feed.selectbox("Feed", list(sim_data.FEEDS))
        col_key, col_secret = st.columns(2)
        api_key = col_key.text_input(
            "Alpaca API key", value=os.getenv("ALPACA_API_KEY", ""), type="password"
        )
        api_secret = col_secret.text_input(
            "Alpaca secret", value=os.getenv("ALPACA_SECRET", ""), type="password"
        )
        st.caption(
            "Alpaca credentials are required for the `iex`/`sip` feeds. On `yfinance` "
            "they are optional and fetch news only — without them the dataset has no news."
        )
        submitted = st.form_submit_button("Download dataset", icon=":material/download:", type="primary")

    if submitted:
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        if not name.strip() or not symbols:
            st.error("A dataset needs a name and at least one symbol.")
        elif feed != "yfinance" and not (api_key and api_secret):
            st.error(f"Alpaca credentials are required for the `{feed}` feed.")
        else:
            with st.status(f"Downloading '{name}'…", expanded=True) as status:
                try:
                    ds = sim_data.create_dataset(
                        name.strip(), symbols, start, end, api_key, api_secret, feed,
                        progress=st.write,
                    )
                    status.update(
                        label=f"Dataset '{ds.name}' ready — {len(ds.days)} trading day(s)",
                        state="complete",
                    )
                except Exception as exc:
                    status.update(label=f"Download failed: {exc}", state="error")

    datasets = sim_data.list_datasets()
    st.divider()
    if not datasets:
        st.info("No datasets yet.")
        return
    size_mb = sim_data.store_size_bytes() / 1e6
    st.markdown(f"##### Stored datasets — shared store {size_mb:.1f} MB")
    for ds in datasets:
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(
                f"**{ds.name}** — {', '.join(ds.symbols)} · {ds.start} → {ds.end} "
                f"· {len(ds.days)} trading day(s) · `{ds.feed}`"
            )
            if st.button("Delete", key=f"del_ds_{ds.name}", icon=":material/delete:"):
                sim_data.delete_dataset(ds.name)
                st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — simulate
# ---------------------------------------------------------------------------

def _equity_chart(equity: list[dict], starting_cash: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p["ts"] for p in equity],
            y=[p["value"] for p in equity],
            mode="lines",
            name="portfolio value",
            line=dict(color=PALETTE["accent"], width=2),
        )
    )
    fig.add_hline(y=starting_cash, line_dash="dot", line_color=PALETTE["muted"])
    return _chart_layout(fig, height=300)


def _price_chart(symbol: str, bars: list[dict], decisions: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=[b["t"] for b in bars],
            open=[b["o"] for b in bars],
            high=[b["h"] for b in bars],
            low=[b["l"] for b in bars],
            close=[b["c"] for b in bars],
            name=symbol,
            increasing_line_color=PALETTE["up"],
            decreasing_line_color=PALETTE["down"],
        )
    )
    for action, color, symbol_marker in (("buy", PALETTE["up"], "triangle-up"), ("sell", PALETTE["down"], "triangle-down")):
        fills = [
            d for d in decisions
            if d.get("symbol") == symbol and d.get("action") == action and d.get("status") == "filled"
        ]
        if fills:
            fig.add_trace(
                go.Scatter(
                    x=[d["ts"] for d in fills],
                    y=[d["price"] for d in fills],
                    mode="markers",
                    name=action,
                    marker=dict(color=color, size=13, symbol=symbol_marker,
                                line=dict(width=1, color=PALETTE["text"])),
                )
            )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _chart_layout(fig, height=420)


def _render_judge_report(judge_report: dict) -> None:
    st.markdown("##### :material/gavel: LLM judge")
    if judge_report.get("judge_model"):
        st.caption(f":material/robot_2: Judged by {judge_report['judge_model']}")
    cols = st.columns(3)
    cols[0].metric("Overall score", f"{judge_report.get('overall_score', '—')}/10")
    cols[1].metric("Strategy adherence", f"{judge_report.get('strategy_adherence', '—')}/10")
    avg = judge_report.get("avg_entry_score")
    cols[2].metric("Avg entry score", f"{avg}/10" if avg is not None else "—")
    if judge_report.get("summary"):
        st.write(judge_report["summary"])
    for item in judge_report.get("top_improvements") or []:
        st.markdown(f"- {item}")
    if judge_report.get("error"):
        st.warning(f"Overall judgment failed: {judge_report['error']}")
    for entry in judge_report.get("entries", []):
        header = f"{entry.get('ts', '')[:16]} · {entry.get('symbol')} @ {entry.get('price')}"
        score = entry.get("score")
        label = f"{header} — {score}/10 ({entry.get('verdict', '?')})" if score is not None else header
        with st.expander(label):
            if entry.get("error"):
                st.warning(entry["error"])
                continue
            st.markdown(f"**Reasoning quality:** {entry.get('reasoning_quality')}")
            st.markdown(f"**What went well:** {entry.get('what_went_well')}")
            st.markdown(f"**To improve:** {entry.get('what_to_improve')}")


def _render_run(record: dict) -> None:
    summary = record.get("summary") or {}
    config = record.get("config_summary") or {}
    if record.get("error"):
        st.error(f"Simulation ended with an error (partial results below): {record['error']}")

    cols = st.columns(6)
    profit = summary.get("profit", 0.0)
    cols[0].metric("Final value", f"${summary.get('final_value', 0):,.0f}",
                   delta=f"{summary.get('return_pct', 0):+.2f}%")
    cols[1].metric("Profit", f"${profit:,.2f}")
    cols[2].metric("Oracle ceiling", f"{summary.get('oracle_ceiling_pct', 0):.2f}%",
                   help="Best single round trip an oracle could have made on this tape.")
    eff = summary.get("profit_efficiency")
    cols[3].metric("Profit efficiency", f"{eff:.1%}" if eff is not None else "—",
                   help="Session return ÷ oracle ceiling.")
    cols[4].metric("Trades filled", summary.get("trades_filled", 0))
    rule_based = bool(config.get("rule_based"))
    cols[5].metric(
        "Bars scored" if rule_based else "LLM cycles",
        summary.get("cycles_run", record.get("cycles_run", 0)),
    )
    if rule_based:
        st.caption(
            f":material/function: Rule-based run — {_agent_label(config.get('personality'))}, "
            f"no LLM and no judge. Rules: `{config.get('model')}`"
        )
    # The tape is part of what produced these numbers: identical rules on
    # `yfinance`, `iex` and `sip` are different bars and can be different
    # trades. A record from before the feed was tracked ran on `iex`.
    feed = config.get("feed") or sim_data.LEGACY_FEED
    st.caption(f":material/database: Tape: `{feed}`")

    equity = record.get("equity") or []
    if equity:
        st.plotly_chart(_equity_chart(equity, record.get("starting_cash", 0.0)))

    symbols = config.get("symbols") or []
    days = [date.fromisoformat(d) for d in config.get("days") or []]
    decisions = record.get("decisions") or []
    if symbols and days:
        try:
            market = SimMarket(symbols, days, feed)
            tabs = st.tabs(symbols)
            for tab, sym in zip(tabs, symbols):
                with tab:
                    bars = market.series[sym].minute_bars
                    if bars:
                        st.plotly_chart(_price_chart(sym, bars, decisions))
                    else:
                        st.info("No stored bars for this symbol/day.")
        except Exception as exc:
            st.caption(f"Price charts unavailable ({exc}).")

    if record.get("judge"):
        _render_judge_report(record["judge"])

    with st.expander(f"Decisions ({len(decisions)})"):
        st.dataframe(
            [
                {k: d.get(k) for k in ("ts", "symbol", "action", "status", "price",
                                       "filled_quantity", "cash_after", "reasoning")}
                for d in decisions
            ],
            height=300,
        )
    log = record.get("agent_log") or []
    with st.expander(f"Agent log ({len(log)} entries)"):
        for entry in log[-400:]:
            ts = str(entry.get("ts", ""))[11:19]
            kind = entry.get("type", "")
            text = entry.get("text") or entry.get("reasoning") or entry.get("name") or ""
            st.markdown(f"`{ts}` **{kind}** {text}")


# The pipeline: queued experiments, a parallelism limit, and worker processes.
# One process = one simulation (module-global sim clock), so parallel runs are
# separate `python -m simlab.runner` subprocesses scheduled by
# `sim_experiments.tick()`; this section is both the status bar and the tick.

MAX_PARALLEL_KEY = "pipeline_max_parallel"

_STATUS_MARKUP = {
    sim_experiments.WAITING: ":orange[:material/hourglass_top: waiting]",
    sim_experiments.RUNNING: ":blue[:material/sync: running]",
    sim_experiments.FINISHED: ":green[:material/check_circle: finished]",
    sim_experiments.FAILED: ":red[:material/error: failed]",
}


def _experiment_label(exp: dict) -> str:
    config = exp.get("config") or {}
    personality = config.get("personality") or "?"
    agent = _agent_label(personality)
    return (
        f"**{agent}** · {config.get('provider')}/{config.get('model')} · "
        f"{exp.get('dataset')} · {len(config.get('days') or [])} day(s)"
    )


def _past_experiment_rows(past: list[dict]) -> list[dict]:
    rows = []
    for exp in past:
        config = exp.get("config") or {}
        result = exp.get("result_summary") or {}
        personality = config.get("personality") or "?"
        rows.append({
            "queued": (exp.get("created_at") or "")[:16].replace("T", " "),
            "agent": _agent_label(personality),
            "model": f"{config.get('provider')}/{config.get('model')}",
            "dataset": exp.get("dataset"),
            "days": len(config.get("days") or []),
            "status": exp.get("status"),
            "return_pct": result.get("return_pct"),
            "profit_efficiency": result.get("profit_efficiency"),
            "judge": result.get("judge_overall"),
            "run_id": exp.get("run_id"),
            "error": exp.get("error"),
        })
    return rows


def _render_pipeline_body(auto_refresh: bool) -> None:
    col_head, col_workers = st.columns([4, 1], vertical_alignment="bottom")
    col_head.markdown("##### :material/stacks: Experiment pipeline")
    max_parallel = col_workers.number_input(
        "Parallel workers", min_value=1, max_value=8, value=2, key=MAX_PARALLEL_KEY,
        help="Experiments beyond this limit wait in the queue. Each experiment "
             "runs in its own worker process.",
    )
    sim_experiments.tick(max_parallel)
    exps = sim_experiments.list_experiments()
    active = [e for e in exps if e["status"] in sim_experiments.ACTIVE_STATUSES]
    if auto_refresh and not active:
        st.rerun()  # pipeline just drained -- refresh the whole app once

    counts = {
        status: sum(1 for e in exps if e["status"] == status)
        for status in (sim_experiments.WAITING, sim_experiments.RUNNING,
                       sim_experiments.FINISHED, sim_experiments.FAILED)
    }
    cols = st.columns(4)
    cols[0].metric("Waiting", counts[sim_experiments.WAITING])
    cols[1].metric("Running", counts[sim_experiments.RUNNING])
    cols[2].metric("Finished", counts[sim_experiments.FINISHED])
    cols[3].metric("Failed", counts[sim_experiments.FAILED])

    for exp in active:
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(_experiment_label(exp))
            st.markdown(_STATUS_MARKUP[exp["status"]])
            if exp["status"] == sim_experiments.RUNNING:
                line = sim_experiments.last_log_line(exp["experiment_id"])
                if line:
                    st.caption(line)
                if st.button(
                    "Stop", key=f"exp_stop_{exp['experiment_id']}",
                    icon=":material/stop_circle:",
                    help="Kill the worker now. The cycles run so far are lost — "
                         "no run record is saved.",
                ):
                    sim_experiments.stop(exp["experiment_id"])
                    st.toast("Experiment stopped", icon=":material/stop_circle:")
                    st.rerun()
            elif st.button(
                "Remove", key=f"exp_rm_{exp['experiment_id']}", icon=":material/close:"
            ):
                sim_experiments.delete_experiment(exp["experiment_id"])
                st.rerun()

    past = [e for e in exps
            if e["status"] in (sim_experiments.FINISHED, sim_experiments.FAILED)]
    if past:
        with st.expander(f"Past experiments ({len(past)})"):
            st.dataframe(
                pd.DataFrame(_past_experiment_rows(past)),
                hide_index=True,
                column_config={
                    "queued": "Queued (UTC)",
                    "agent": "Agent",
                    "model": "Model",
                    "dataset": "Dataset",
                    "days": "Days",
                    "status": "Status",
                    "return_pct": st.column_config.NumberColumn("Return", format="%+.2f%%"),
                    "profit_efficiency": st.column_config.NumberColumn(
                        "Profit eff.", format="percent"),
                    "judge": st.column_config.NumberColumn("Judge", format="%.1f"),
                    "run_id": "Run",
                    "error": "Error",
                },
            )
            if st.button("Clear history", icon=":material/delete_sweep:",
                         help="Drops the experiment records; stored runs are kept."):
                sim_experiments.clear_finished()
                st.rerun()


def _render_pipeline() -> None:
    # While experiments are active the section refreshes itself (which also
    # ticks the scheduler); once idle it renders statically until the next
    # full rerun.
    auto_refresh = sim_experiments.has_active()
    st.fragment(run_every=2.5 if auto_refresh else None)(
        lambda: _render_pipeline_body(auto_refresh)
    )()


def _prior_run_line(record: dict, selected_days: list[str]) -> str:
    summary = record.get("summary") or {}
    judge = record.get("judge") or {}
    run_days = (record.get("config_summary") or {}).get("days") or []
    parts = [
        f"`{record['run_id']}`",
        ", ".join(run_days) or "—",
        f"{summary.get('return_pct', 0):+.2f}%",
    ]
    eff = summary.get("profit_efficiency")
    if eff is not None:
        parts.append(f"eff {eff:.1%}")
    if judge.get("overall_score") is not None:
        parts.append(f"judge {judge['overall_score']}/10")
    line = " · ".join(parts)
    if selected_days and set(run_days) == set(selected_days):
        line += " — :orange[**same trading day(s)**]"
    return line


def _combo_label(combo: tuple[str, str, str, str]) -> str:
    personality, provider, model, dataset = combo
    return (
        f"**{_agent_label(personality)}** · "
        f"{provider}/{model} · {dataset}"
    )


def _render_already_tested(
    combos: list[tuple[str, str, str, str]], days_by_dataset: dict[str, list[str]]
) -> set[tuple[str, str, str, str]]:
    """Flag agent/model/dataset combinations that are already queued or have
    already been tested cleanly, so the same experiment isn't paid for twice.
    Runs whose cycles hit LLM errors don't count as tested -- those are exactly
    the ones worth running again. Returns the cleanly tested combinations."""
    pending: dict[tuple[str, str, str, str], list[dict]] = {}
    for exp in sim_experiments.list_experiments():
        if exp["status"] not in sim_experiments.ACTIVE_STATUSES:
            continue
        config = exp.get("config") or {}
        key = (config.get("personality"), config.get("provider"),
               config.get("model"), exp.get("dataset"))
        pending.setdefault(key, []).append(exp)

    runs = _runs()
    queued_lines, tested, tested_lines, degraded_lines = [], set(), [], []
    for combo in combos:
        personality, provider, model, dataset = combo
        if combo in pending:
            statuses = ", ".join(sorted({e["status"] for e in pending[combo]}))
            queued_lines.append(
                f"- {_combo_label(combo)} — {len(pending[combo])} experiment(s): {statuses}"
            )
        prior = sim_results.find_prior_runs(runs, personality, provider, model, dataset)
        clean, degraded = prior["clean"], prior["degraded"]
        if clean:
            tested.add(combo)
            days = days_by_dataset.get(dataset) or []
            runs_text = "\n".join(
                f"    - {_prior_run_line(r, days)}" for r in clean[:5]
            )
            more = f"\n    - …and {len(clean) - 5} more" if len(clean) > 5 else ""
            tested_lines.append(
                f"- {_combo_label(combo)} — {len(clean)} clean run(s)\n{runs_text}{more}"
            )
        elif degraded:
            errors = sum(sim_results.cycle_error_count(r) for r in degraded)
            degraded_lines.append(
                f"- {_combo_label(combo)} — run {len(degraded)} time(s) before, but "
                f"{errors} cycle(s) failed (LLM errors)"
            )

    if queued_lines:
        st.warning(
            f":material/schedule: {len(queued_lines)} combination(s) already in the "
            "pipeline:\n" + "\n".join(queued_lines)
        )
    if tested_lines:
        st.warning(
            f":material/history: {len(tested_lines)} combination(s) have **already been "
            "tested** cleanly — results are in the Results tab. Re-run only if you "
            "changed the prompt or the settings below.\n" + "\n".join(tested_lines)
        )
    if degraded_lines:
        st.info(
            ":material/error_outline: These combinations ran before but hit LLM errors, "
            "so they aren't a clean test:\n" + "\n".join(degraded_lines)
        )
    return tested


def _render_dataset_scope(datasets: list) -> dict[str, dict]:
    """Per-dataset trading days and symbols -- each selected dataset carries its
    own, since days and symbols differ from dataset to dataset."""
    scope: dict[str, dict] = {}
    for ds in datasets:
        with st.container(border=True):
            st.markdown(f"**{ds.name}** — {ds.start} → {ds.end} · tape `{ds.feed}`")
            col_days, col_symbols = st.columns(2)
            day_options = ds.days or []
            scope[ds.name] = {
                "days": col_days.multiselect(
                    "Trading day(s)", day_options, default=day_options[:1],
                    key=f"sim_days_{ds.name}",
                ),
                "symbols": col_symbols.multiselect(
                    "Symbols", ds.symbols, default=ds.symbols, key=f"sim_syms_{ds.name}",
                ),
            }
    return scope


def _render_model_picker() -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Models to test, across providers: returns [(provider, model), …] and the
    API key per selected provider."""
    providers = st.pills(
        "Providers", list(PROVIDERS), selection_mode="multi",
        default=[PROVIDERS[0]], key="sim_providers",
    ) or []
    model_choices: list[tuple[str, str]] = []
    api_keys: dict[str, str] = {}
    for provider in providers:
        with st.container(border=True):
            col_models, col_key = st.columns([2, 1])
            picked = col_models.multiselect(
                f"{provider} models",
                models_for(provider, default=DEFAULT_AGENT_MODELS[provider]),
                default=[DEFAULT_AGENT_MODELS[provider]],
                key=f"sim_models_{provider}",
            )
            api_keys[provider] = col_key.text_input(
                f"{provider} API key", value=_env_key(provider), type="password",
                key=f"sim_key_{provider}",
            )
            model_choices += [(provider, model) for model in picked]
    return model_choices, api_keys


def _rule_agents_missing_ticker(
    personalities: list[str], dataset_scope: dict, selected_names: list[str]
) -> dict[str, list[str]]:
    """Rule agent -> the selected datasets that do not carry its one symbol.

    A dataset without it is not a strategy result, it is a run that cannot
    trade, so it is worth catching before the experiments are queued.
    """
    missing: dict[str, list[str]] = {}
    for personality in personalities:
        ticker = rule_agent(personality).ticker
        names = [
            name for name in selected_names
            if ticker not in (dataset_scope[name]["symbols"] or [])
        ]
        if names:
            missing[personality] = names
    return missing


def _render_rule_params(personalities: list[str]) -> dict:
    """One rule set per selected rule agent, the way one prompt per personality
    applies to every LLM combination.

    Keyed by personality, since the two rule agents share no tunables at all --
    the returned configs are what the queued experiments carry.
    """
    renderers = {
        APPLE_TRADER_KEY: _render_apple_params,
        TRADER_BY_CHATGPT_KEY: _render_chatgpt_params,
        TRADER_BY_CLAUDE_KEY: _render_claude_params,
    }
    return {key: renderers[key]() for key in personalities if key in renderers}


def _render_claude_params() -> TraderByClaudeConfig:
    """TraderByClaude's rules for this batch.

    Exposed are the four numbers that change what the strategy *is*: how deep a
    pullback it will look back for, how quiet that pullback has to be, how far
    from the mean it will still buy, and how much equity it risks per trade.
    The EMA pair, the entry window and the exits are the strategy's definition
    rather than knobs, and stay at their defaults.
    """
    defaults = TraderByClaudeConfig()
    with st.expander("TraderByClaude rules", expanded=True):
        st.caption(
            "The counter-thesis to TraderByChatGPT: buy the pullback into the trend, "
            "not the breakout out of it. Each distinct rule set is tracked as its own "
            "configuration in Results, so retuning is a new test rather than a repeat "
            "of one already run."
        )
        col_a, col_b = st.columns(2)
        pullback_lookback = col_a.number_input(
            "Pullback window (bars)", min_value=3, max_value=60,
            value=defaults.pullback_lookback, step=1, key="sim_claude_pullback",
            help="How far back the dip to the mean may have happened. Longer accepts "
                 "setups whose pullback is older, and puts the structural stop further "
                 "away, since the stop goes under the lowest low in this window.",
        )
        quiet_pullback_ratio = col_b.number_input(
            "Pullback volume (× median)", min_value=0.25, max_value=3.0,
            value=defaults.quiet_pullback_ratio, step=0.05, format="%.2f",
            key="sim_claude_quiet",
            help="Average volume across the pullback, as a multiple of the 20-bar "
                 "median. Below 1.0 means participation dried up on the dip — which is "
                 "what separates a pullback from distribution. Raising it above 1.0 "
                 "removes the test that does the most filtering.",
        )
        max_stretch_atr = col_a.number_input(
            "Stretch limit (ATRs above the mean)", min_value=0.1, max_value=4.0,
            value=defaults.max_stretch_atr, step=0.05, format="%.2f",
            key="sim_claude_stretch",
            help="The anti-chase rule: refuse a reclaim bar that has already run this "
                 "far past the mean. This binds against the reclaim test — a bar that "
                 "takes out the prior high tends to close some way above the mean — so "
                 "it is the main control on how often the agent trades at all.",
        )
        risk_pct = col_b.number_input(
            "Risk per trade (% of equity)", min_value=0.05, max_value=5.0,
            value=defaults.risk_pct, step=0.05, format="%.2f", key="sim_claude_risk",
            help="Dollars between the fill and the structural stop, as a share of "
                 "equity. Because that stop sits under the pullback low rather than a "
                 "fixed ATR away, the same percentage usually buys a larger position "
                 "here than in a breakout entry.",
        )
        max_position_pct = col_a.number_input(
            "Position cap (% of cash)", min_value=1.0, max_value=100.0,
            value=defaults.max_position_pct, step=5.0, key="sim_claude_size",
        )
    return TraderByClaudeConfig(
        pullback_lookback=int(pullback_lookback),
        quiet_pullback_ratio=float(quiet_pullback_ratio),
        max_stretch_atr=float(max_stretch_atr),
        risk_pct=float(risk_pct),
        max_position_pct=float(max_position_pct),
    )


def _render_chatgpt_params() -> TraderByChatGPTConfig:
    """TraderByChatGPT's rules for this batch.

    The entry is a confluence of seven conditions and the exit is an ATR trail;
    exposed here are the four that change what the strategy *is* -- how far
    back the breakout looks, how much volume confirmation it demands, how much
    equity it risks per trade, and how far the stop trails. The rest (the RSI
    band, the ATR regime filter, the entry window) are the strategy's
    definition rather than knobs, and stay at their defaults.
    """
    defaults = TraderByChatGPTConfig()
    with st.expander("TraderByChatGPT rules", expanded=True):
        st.caption(
            "No model to tune — the entry is a plain boolean over the bar that just "
            "closed, so these are the numbers inside it. Each distinct rule set is "
            "tracked as its own configuration in Results, so retuning is a new test "
            "rather than a repeat of one already run."
        )
        col_a, col_b = st.columns(2)
        breakout_lookback = col_a.number_input(
            "Breakout window (bars)", min_value=5, max_value=120,
            value=defaults.breakout_lookback, step=5, key="sim_chatgpt_lookback",
            help="The bar has to close above the highest high of this many bars before "
                 "it. Shorter fires more often on smaller ranges.",
        )
        min_relative_volume = col_b.number_input(
            "Relative volume to confirm", min_value=1.0, max_value=5.0,
            value=defaults.min_relative_volume, step=0.1, format="%.2f",
            key="sim_chatgpt_rvol",
            help="Volume as a multiple of its 20-bar median. A breakout without unusual "
                 "participation is the one that fails back into the range.",
        )
        risk_pct = col_a.number_input(
            "Risk per trade (% of equity)", min_value=0.05, max_value=5.0,
            value=defaults.risk_pct, step=0.05, format="%.2f", key="sim_chatgpt_risk",
            help="Dollars between the fill and the initial stop, as a share of equity. "
                 "This sizes the position — the position cap only stops a tiny ATR from "
                 "asking for an unbounded one.",
        )
        trail_atr = col_b.number_input(
            "Trailing stop (ATRs under the peak)", min_value=0.25, max_value=5.0,
            value=defaults.trail_atr, step=0.25, format="%.2f", key="sim_chatgpt_trail",
            help="Once the trade is 1R ahead the stop ratchets up to this far under the "
                 "highest price seen since the entry. Before that the initial "
                 f"{defaults.initial_stop_atr:g} ATR stop stands.",
        )
        max_position_pct = col_a.number_input(
            "Position cap (% of cash)", min_value=1.0, max_value=100.0,
            value=defaults.max_position_pct, step=5.0, key="sim_chatgpt_size",
        )
        min_atr_pct = col_b.number_input(
            "Minimum ATR (% of price)", min_value=0.0, max_value=1.0,
            value=defaults.min_atr_pct * 100.0, step=0.01, format="%.3f",
            key="sim_chatgpt_atr",
            help="Refuse to buy a breakout on tape too quiet to travel anywhere. These "
                 "are one-minute bars, where a liquid large cap runs a median ATR near "
                 "0.08% of price — two orders of magnitude under the daily figure, so a "
                 "daily-scale floor here rejects every bar of the session.",
        )
    return TraderByChatGPTConfig(
        breakout_lookback=int(breakout_lookback),
        min_relative_volume=float(min_relative_volume),
        risk_pct=float(risk_pct),
        trail_atr=float(trail_atr),
        max_position_pct=float(max_position_pct),
        min_atr_pct=float(min_atr_pct) / 100.0,
    )


def _render_apple_params() -> AppleTraderConfig:
    """Apple Trader's rules for this batch."""
    defaults = AppleTraderConfig()
    with st.expander("Apple Trader rules", expanded=True):
        st.caption(
            "Four knobs decide everything: **when** the saved model is asked about a "
            "regime change, which model is asked, how sure it has to be, and how much of "
            "the run the trade gives back before selling. Each distinct rule set is "
            "tracked as its own configuration in Results, so moving the entry, swapping "
            "the model or retuning the stop is a new test rather than a repeat of one "
            "already run."
        )
        entry_mode = st.segmented_control(
            "Entry", ENTRY_MODES, default=defaults.entry_mode,
            format_func=lambda mode: ENTRY_MODE_LABEL.get(mode, mode), key="sim_apple_entry_mode",
            help="The setting that moves the fill most. On the 2026-07-27 SIP tape "
                 "“Confirm” bought 337.45 / 338.67 / 336.35 and “Anticipate” bought the "
                 "same three episodes at 336.56 / 338.20 / 335.99 — one to six bars "
                 "earlier, while the regime was still balanced, taking the session from "
                 "−0.41% to +0.08%. That is three trades on one day: a check that the "
                 "wiring works, not a measurement of the edge.",
        ) or defaults.entry_mode
        st.caption(ENTRY_MODE_SUMMARY[entry_mode])

        keys = apple_models.keys()
        model_key = st.selectbox(
            "Model", keys,
            index=keys.index(defaults.model_key) if defaults.model_key in keys else 0,
            format_func=lambda key: apple_models.get(key).label
            + ("" if apple_models.get(key).anticipates else " — cannot anticipate"),
            key="sim_apple_model",
            help="On the confirm entry both models are handed the same 20 bars on the "
                 "same tape and return one probability, so running a dataset through "
                 "both is a straight comparison. Only the forecaster can run the "
                 "anticipate entry at all.",
        )
        model = apple_models.get(model_key)
        bundle = apple_models.load(model.key)
        st.caption(model.summary)
        if bundle is None:
            st.error(apple_models.unavailable_reason(model_key))
        elif entry_mode == ENTRY_ANTICIPATE and not model.anticipates:
            st.error(
                f"{model.label} was fitted on regime-change bars only, so it cannot "
                "forecast a change that has not happened yet. Pick a forecasting model "
                "or switch the entry to “Confirm the turn”; this pairing stops the run "
                "rather than producing an empty ledger."
            )
        bundle_threshold = persistence_model.model_threshold(bundle)

        col_a, col_b = st.columns(2)
        prob_threshold = col_a.number_input(
            ENTRY_MODE_PROB_LABEL[entry_mode], min_value=0.0, max_value=1.0,
            value=float(defaults.prob_threshold or bundle_threshold),
            step=0.01, format="%.2f",
            # Keyed by model so switching re-seeds the input with that model's
            # own cut-off: the two probabilities are not on a shared scale.
            key=f"sim_apple_prob_{model_key}",
            help=f"Default {bundle_threshold:g} is the cut-off this model chose on its own "
                 "validation block — on the *confirm* question. On “Anticipate” it is a "
                 "starting point rather than a tuned setting, and it is the first thing "
                 "worth sweeping here: it decides how early in the build-up the entry "
                 "fires.",
        )
        trail_pct = col_b.number_input(
            "Trailing stop (%)", min_value=0.05, max_value=10.0,
            value=defaults.trail_pct, step=0.05, key="sim_apple_trail",
            help="Sell once price is this far below the highest price seen since the "
                 "entry. The peak only ratchets up, so this starts as a stop under the "
                 "entry and becomes a profit lock as the move runs.",
        )
        position_pct = col_a.number_input(
            "Position size (% of cash)", min_value=1.0, max_value=100.0,
            value=defaults.position_pct, step=5.0, key="sim_apple_size",
        )
    return AppleTraderConfig(
        model_key=str(model_key),
        entry_mode=str(entry_mode),
        prob_threshold=float(prob_threshold),
        trail_pct=float(trail_pct),
        position_pct=float(position_pct),
    )


def render_simulate_tab() -> None:
    _render_pipeline()
    st.divider()

    datasets = sim_data.list_datasets()
    if not datasets:
        st.info("Download a dataset first (Datasets tab).")
        return

    st.caption(
        "Pick several agents, models, and datasets — every combination is queued as "
        "its own experiment."
    )
    names = [d.name for d in datasets]
    selected_names = st.multiselect(
        "Datasets", names, default=names[:1], key="sim_datasets"
    )
    by_name = {d.name: d for d in datasets}
    dataset_scope = _render_dataset_scope([by_name[name] for name in selected_names])
    personalities = st.multiselect(
        "Agents", _testable_agents(),
        default=_testable_agents()[:1],
        format_func=_agent_label,
        key="sim_agents",
    )
    llm_personalities = [p for p in personalities if sim_prompts.has_prompt(p)]
    rule_personalities = [p for p in personalities if not sim_prompts.has_prompt(p)]
    rule_configs = _render_rule_params(rule_personalities)
    # The model picker only sizes the LLM grid: a rule agent runs the same
    # way whatever is selected there, so it is queued once per dataset instead.
    model_choices, api_keys = (
        _render_model_picker() if llm_personalities else ([], {})
    )

    with st.expander("Simulation settings"):
        col_cash, col_cycle, col_max = st.columns(3)
        starting_cash = col_cash.number_input("Starting cash", value=100_000.0, step=10_000.0)
        cycle_minutes = col_cycle.number_input(
            "Cycle interval (min)", value=5, min_value=1, max_value=60,
            help="Re-cycle cadence while nothing is armed. With alerts/tactics armed the "
                 "agent sleeps until a condition fires, exactly as live.",
        )
        max_cycles = col_max.number_input("Max LLM cycles per day", value=40, min_value=1, max_value=200)
        run_judge = st.checkbox(
            "Judge the run with an LLM after the simulation", value=True,
            help="Grades every entry on the information available at entry time, plus an "
                 "overall strategy-adherence review.",
        )
        if run_judge and rule_personalities:
            st.caption(
                ":material/gavel: "
                + ", ".join(_agent_label(p) for p in rule_personalities)
                + (" are" if len(rule_personalities) > 1 else " is")
                + " never judged — they state no reasoning of their own, so profit, "
                "profit efficiency and the oracle ceiling are the whole scorecard."
            )
        # Judging with the agent's own model is per-combination; a single
        # explicit judge is shared by every experiment in the grid.
        judge_override = run_judge and bool(llm_personalities) and st.checkbox(
            "Judge with a different LLM than the agent", value=False,
            help="By default each agent's own provider/model grades its run. Pick a "
                 "separate judge to avoid an agent marking its own homework.",
        )
        judge_provider = judge_model = judge_api_key = None
        if judge_override:
            col_jprov, col_jmodel, col_jkey = st.columns(3)
            judge_provider = col_jprov.selectbox("Judge provider", list(PROVIDERS))
            judge_model = col_jmodel.selectbox(
                "Judge model",
                models_for(judge_provider, default=DEFAULT_AGENT_MODELS[judge_provider]),
            )
            judge_api_key = col_jkey.text_input(
                "Judge API key", value=_env_key(judge_provider), type="password"
            )

    # One combination per (agent, model, dataset) for the LLM agents; a rule
    # agent has no model dimension, so its own rule set stands in for one --
    # which also means retuning it queues a genuinely new combination.
    rule_signatures = {
        key: rule_agent(key).signature(config) for key, config in rule_configs.items()
    }
    combos = [
        (personality, provider, model, name)
        for name in selected_names
        for personality in llm_personalities
        for provider, model in model_choices
    ] + [
        (personality, RULE_PROVIDER, rule_signatures[personality], name)
        for name in selected_names
        for personality in rule_personalities
    ]
    days_by_dataset = {name: scope["days"] for name, scope in dataset_scope.items()}
    tested = _render_already_tested(combos, days_by_dataset)
    skip_tested = bool(tested) and st.checkbox(
        f"Skip the {len(tested)} combination(s) already tested cleanly", value=True,
        key="sim_skip_tested",
        help="Uncheck to run them again anyway — e.g. after editing a prompt.",
    )

    overridden = [p for p in personalities if sim_prompts.has_override(p)]
    if overridden:
        labels = ", ".join(_agent_label(p) for p in overridden)
        st.caption(f":material/edit: Runs with a **modified** prompt (Agents tab): {labels}.")
    missing_ticker = _rule_agents_missing_ticker(rule_personalities, dataset_scope,
                                                 selected_names)
    for personality, names_missing in missing_ticker.items():
        st.error(
            f":material/error: {_agent_label(personality)} only trades "
            f"{rule_agent(personality).ticker}, which is not selected for: "
            f"{', '.join(names_missing)}."
        )
    st.caption(
        ":material/monitoring: Langfuse export: "
        + ("enabled — cycles are traced and run scores registered." if obs.is_enabled()
           else "disabled (set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to enable).")
    )

    to_queue = [c for c in combos if not (skip_tested and c in tested)]
    with st.container(horizontal=True, vertical_alignment="center"):
        run = st.button(
            f"Run {len(to_queue)} experiment(s)", icon=":material/play_arrow:",
            type="primary", disabled=not to_queue,
            help="Adds every combination to the pipeline; each starts as soon "
                 "as a worker slot is free.",
        )
        if combos:
            parts = []
            if llm_personalities:
                parts.append(
                    f"{len(llm_personalities)} LLM agent(s) × {len(model_choices)} model(s)"
                )
            if rule_personalities:
                parts.append(f"{len(rule_personalities)} rule-based agent(s)")
            st.caption(
                f"({' + '.join(parts)}) × {len(selected_names)} dataset(s) = "
                f"{len(combos)} combination(s)"
                + (f", {len(combos) - len(to_queue)} skipped" if skip_tested else "")
            )
        elif personalities and not llm_personalities:
            st.caption("Pick at least one dataset.")
        else:
            st.caption("Pick at least one dataset, agent, and model.")

    if run:
        empty = [
            name for name in selected_names
            if not dataset_scope[name]["days"] or not dataset_scope[name]["symbols"]
        ]
        missing_keys = sorted({provider for provider, _ in model_choices
                               if not api_keys.get(provider)})
        if empty:
            st.error(
                "Pick at least one trading day and one symbol for: " + ", ".join(empty)
            )
        elif missing_ticker:
            personality, names_missing = next(iter(missing_ticker.items()))
            st.error(
                f"Add {rule_agent(personality).ticker} to the symbols of "
                f"{', '.join(names_missing)}, or deselect "
                f"{_agent_label(personality)}."
            )
        elif missing_keys:
            st.error(f"An API key is required for: {', '.join(missing_keys)}.")
        elif run_judge and judge_override and not judge_api_key:
            st.error(f"An API key for the judge provider ({judge_provider}) is required.")
        else:
            for personality, provider, model, name in to_queue:
                scope = dataset_scope[name]
                rule_based = personality in rule_personalities
                sim_experiments.submit(name, {
                    "personality": personality,
                    "provider": provider,
                    "model": model,
                    "api_key": "" if rule_based else api_keys[provider],
                    "symbols": scope["symbols"],
                    "days": scope["days"],
                    "starting_cash": float(starting_cash),
                    "cycle_minutes": int(cycle_minutes),
                    "max_cycles_per_day": int(max_cycles),
                    "feed": by_name[name].feed,
                    "system_prompt_override": sim_prompts.get_override(personality),
                    "rule_config": (
                        rule_agent(personality).to_record(rule_configs[personality])
                        if rule_based else None
                    ),
                    # A rule agent is never judged: no reasoning of its own to
                    # grade, so the profit metrics are the whole scorecard.
                    "run_judge": bool(run_judge) and not rule_based,
                    "judge_provider": judge_provider or provider,
                    "judge_model": judge_model or model,
                    "judge_api_key": judge_api_key or api_keys.get(provider, ""),
                })
            sim_experiments.tick(int(st.session_state.get(MAX_PARALLEL_KEY, 2)))
            st.toast(f"{len(to_queue)} experiment(s) queued",
                     icon=":material/rocket_launch:")
            st.rerun()


# ---------------------------------------------------------------------------
# Tabs 4 & 5 — summary (aggregates) and results (one run at a time)
# ---------------------------------------------------------------------------

# Agent first and by default: it is the thing under test. "Model" covers both
# the LLM behind a personality and the rule set behind the rule-based agent --
# both are what varies while agent and dataset are held fixed.
_BREAKDOWN_DIMENSIONS = {"Agent": "agent", "Model": "model", "Dataset": "dataset"}

# Ranking metrics for the top-runs cards, mapped to their `summary` keys.
_TOP_RUN_METRICS = {"Best return": "return_pct", "Profit efficiency": "profit_efficiency"}


# The breakdown table is hand-rolled HTML rather than st.dataframe: only a real
# `title` attribute gives the best-return cell a hover tooltip naming the run
# behind it (st.column_config's `help` only tooltips the column header).
_BREAKDOWN_CSS = f"""
<style>
table.simlab-breakdown {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
table.simlab-breakdown th, table.simlab-breakdown td {{
    padding: 0.4rem 0.6rem; text-align: right; white-space: nowrap;
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
}}
table.simlab-breakdown th {{ font-weight: 600; opacity: 0.75; }}
table.simlab-breakdown th:first-child, table.simlab-breakdown td:first-child {{ text-align: left; }}
table.simlab-breakdown th[title] {{ cursor: help; text-decoration: underline dotted; }}
table.simlab-breakdown .best {{ cursor: help; text-decoration: underline dotted; }}
table.simlab-breakdown .up {{ color: {PALETTE["up"]}; }}
table.simlab-breakdown .down {{ color: {PALETTE["down"]}; }}
table.simlab-breakdown .none {{ opacity: 0.45; }}
</style>
"""


def _best_run_tooltip(best: "dict | None") -> str:
    """Which model / agent / dataset produced a group's best return."""
    if not best:
        return ""
    provider, model = best.get("provider"), best.get("model")
    lines = [
        f"Model: {'/'.join(p for p in (provider, model) if p) or '?'}",
        "Agent: " + _agent_label(best.get("personality")),
        f"Dataset: {best.get('dataset') or '(no dataset)'}",
    ]
    if best.get("run_id"):
        lines.append(f"Run: {best['run_id']}")
    return "\n".join(lines)


def _render_breakdown_table(rows: list[dict], dim_label: str) -> None:
    headers = [
        (dim_label, ""),
        ("Runs", ""),
        ("Avg return", ""),
        ("Best return", "Best single run in this group — hover the value for its "
                        "model, agent, and dataset."),
        ("Avg profit efficiency",
         "Session return ÷ oracle best-round-trip ceiling, averaged over runs."),
        ("Avg judge score", "LLM judge overall score (0–10), averaged over judged runs."),
    ]
    head = "".join(
        f'<th title="{escape(help_text, quote=True)}">{escape(label)}</th>'
        if help_text else f"<th>{escape(label)}</th>"
        for label, help_text in headers
    )
    missing = '<span class="none">—</span>'

    def _pct(value: "float | None") -> str:
        if value is None:
            return missing
        return f'<span class="{"up" if value >= 0 else "down"}">{value:+.2f}%</span>'

    body = []
    for row in rows:
        best = _pct(row["best_return_pct"])
        tooltip = _best_run_tooltip(row.get("best_run"))
        if tooltip and row["best_return_pct"] is not None:
            # &#10; keeps the tooltip multi-line without a raw newline inside the
            # attribute, which markdown would treat as a block break.
            title = escape(tooltip, quote=True).replace("\n", "&#10;")
            best = f'<span class="best" title="{title}">{best}</span>'
        efficiency = row["avg_profit_efficiency"]
        score = row["avg_judge_score"]
        body.append(
            "<tr>"
            f"<td>{escape(str(row['group']))}</td>"
            f"<td>{row['runs']}</td>"
            f"<td>{_pct(row['avg_return_pct'])}</td>"
            f"<td>{best}</td>"
            f"<td>{f'{efficiency:.1%}' if efficiency is not None else missing}</td>"
            f"<td>{f'{score:.1f}' if score is not None else missing}</td>"
            "</tr>"
        )
    st.markdown(
        _BREAKDOWN_CSS
        + f'<table class="simlab-breakdown"><thead><tr>{head}</tr></thead>'
        + f"<tbody>{''.join(body)}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption("Hover a best return to see the run behind it.")


def _breakdown_bar(labels: list[str], values: list[float], title: str, color: str,
                   tickformat: "str | None" = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=color))
    fig = _chart_layout(fig, height=320)
    fig.update_layout(title=title, showlegend=False)
    if tickformat:
        fig.update_yaxes(tickformat=tickformat)
    return fig


def _render_run_filters(runs: list[dict], key_prefix: str) -> list[dict]:
    """Dataset / model filters over the stored runs. Selecting nothing in a
    filter leaves that dimension unrestricted, so the default view is all runs.
    Everything below (breakdown, charts, run picker) works off the result.
    Summary and Results each render their own copy -- hence the key prefix --
    so filtering one tab doesn't silently reshape the other."""
    options = sim_results.filter_options(runs)
    col_datasets, col_models = st.columns(2)
    datasets = col_datasets.multiselect(
        "Datasets", options["datasets"], key=f"{key_prefix}_filter_datasets",
        placeholder="All datasets",
    )
    models = col_models.multiselect(
        "Models", options["models"], key=f"{key_prefix}_filter_models",
        placeholder="All models",
    )
    filtered = sim_results.filter_runs(runs, datasets=datasets, models=models)
    if datasets or models:
        st.caption(f"Showing {len(filtered)} of {len(runs)} runs.")
    return filtered


def _short_model(key: str, limit: int = 46) -> str:
    """Rule agents encode their entire rule set in the model string, which would
    otherwise stretch one card far past the others. The full string is still in
    the breakdown table and on the run itself."""
    return key if len(key) <= limit else key[: limit - 1].rstrip() + "…"


def _render_top_runs(runs: list[dict]) -> None:
    """The three best single runs under the filters, on whichever metric is
    picked. Return and profit efficiency disagree often -- a big return on an
    easy tape can be a worse trade than a small one on a flat tape -- so both
    are always shown, only the ranking changes."""
    st.markdown("##### Top runs")
    metric_label = st.segmented_control(
        "Rank by", list(_TOP_RUN_METRICS), default="Best return",
        key="summary_top_metric",
    ) or "Best return"
    metric = _TOP_RUN_METRICS[metric_label]
    top = sim_results.top_runs(runs, by=metric, limit=3)
    if not top:
        st.caption(f"No runs scored on {metric_label.lower()} yet.")
        return
    for rank, (column, row) in enumerate(zip(st.columns(len(top)), top), start=1):
        efficiency = row["profit_efficiency"]
        return_pct = row["return_pct"]
        headline = (
            f"{return_pct:+.2f}%" if metric == "return_pct"
            else f"{efficiency:.1%}"
        )
        if metric == "return_pct":
            secondary = (
                f"Profit efficiency {efficiency:.1%}" if efficiency is not None
                else "Profit efficiency —"
            )
        else:
            secondary = (
                f"Return {return_pct:+.2f}%" if return_pct is not None
                else "Return —"
            )
        with column.container(border=True):
            st.metric(f"#{rank} · {_agent_label(row['personality'])}", headline)
            st.caption(
                f"{secondary}  \n{_short_model(row['model'])} · {row['dataset']}"
                f"  \n`{row['run_id']}`"
            )


@st.dialog("Delete all runs?")
def _confirm_delete_all_runs(total: int) -> None:
    """Wiping the store is irreversible and the filters make it easy to forget
    how much is actually in there, so the count is spelled out before it goes."""
    st.markdown(
        f"This permanently deletes **all {total} stored run"
        f"{'' if total == 1 else 's'}**, including any hidden by the current "
        "filters. Scores already exported to Langfuse are kept."
    )
    with st.container(horizontal=True):
        if st.button("Delete them", type="primary", icon=":material/delete_forever:"):
            removed = sim_results.delete_all_runs()
            for key in (
                "last_run_id",
                "results_filter_datasets", "results_filter_models",
                "summary_filter_datasets", "summary_filter_models",
            ):
                st.session_state.pop(key, None)
            st.toast(f"Deleted {removed} run{'' if removed == 1 else 's'}",
                     icon=":material/delete_sweep:")
            st.rerun()
        if st.button("Cancel", icon=":material/close:"):
            st.rerun()


def render_summary_tab() -> None:
    all_runs = _runs()
    if not all_runs:
        st.info("No stored runs yet — queue an experiment in the Simulate tab.")
        return

    runs = _render_run_filters(all_runs, "summary")
    if not runs:
        st.info("No runs match the current filters.")
        return

    _render_top_runs(runs)

    st.divider()
    st.markdown("##### Breakdown")
    dim_label = st.segmented_control(
        "Break down by", list(_BREAKDOWN_DIMENSIONS), default="Agent",
        key="results_breakdown_dim",
    ) or "Agent"
    dimension = _BREAKDOWN_DIMENSIONS[dim_label]
    rows = sim_results.breakdown(runs, dimension)
    if dimension == "agent":
        for row in rows:
            row["group"] = _agent_label(row["group"])
    _render_breakdown_table(rows, dim_label)
    col_eff, col_score = st.columns(2)
    eff_rows = [r for r in rows if r["avg_profit_efficiency"] is not None]
    if eff_rows:
        col_eff.plotly_chart(_breakdown_bar(
            [r["group"] for r in eff_rows],
            [r["avg_profit_efficiency"] for r in eff_rows],
            "Avg profit efficiency", PALETTE["accent"], tickformat=".0%",
        ))
    score_rows = [r for r in rows if r["avg_judge_score"] is not None]
    if score_rows:
        col_score.plotly_chart(_breakdown_bar(
            [r["group"] for r in score_rows],
            [r["avg_judge_score"] for r in score_rows],
            "Avg judge score (0–10)", PALETTE["up"],
        ))


def render_results_tab() -> None:
    all_runs = _runs()
    if not all_runs:
        st.info("No stored runs yet — queue an experiment in the Simulate tab.")
        return

    runs = _render_run_filters(all_runs, "results")
    if not runs:
        st.info("No runs match the current filters.")
        return

    labels = {
        r["run_id"]: (
            f"{r['run_id']} · {r.get('config_summary', {}).get('personality')} · "
            f"{r.get('config_summary', {}).get('model')} · {r.get('dataset')} · "
            f"{r.get('summary', {}).get('return_pct', 0):+.2f}%"
        )
        for r in runs
    }
    default_id = st.session_state.get("last_run_id", runs[0]["run_id"])
    ids = list(labels)
    selected = st.selectbox(
        "Run", ids,
        index=ids.index(default_id) if default_id in ids else 0,
        format_func=lambda rid: labels[rid],
    )
    record = next(r for r in runs if r["run_id"] == selected)
    _render_run(record)
    with st.container(horizontal=True):
        if st.button("Delete this run", icon=":material/delete:"):
            sim_results.delete_run(selected)
            st.session_state.pop("last_run_id", None)
            st.rerun()
        if st.button(
            "Delete all runs", icon=":material/delete_sweep:",
            help="Clears the whole run store, not just the filtered runs.",
        ):
            _confirm_delete_all_runs(len(all_runs))


# ---------------------------------------------------------------------------

def build_ui() -> None:
    st.set_page_config(page_title="AgentStonks SimLab", page_icon="🧪", layout="wide")
    st.title("SimLab — strategy testing")
    st.caption(
        "Replay the trading agents against stored historical sessions: same prompts, same "
        "tools, same execution path as live — hours of tape in minutes of simulation."
    )
    tab_agents, tab_datasets, tab_sim, tab_summary, tab_results = st.tabs(
        [":material/smart_toy: Agents", ":material/database: Datasets",
         ":material/play_circle: Simulate", ":material/leaderboard: Summary",
         ":material/insights: Results"]
    )
    with tab_agents:
        render_agents_tab()
    with tab_datasets:
        render_datasets_tab()
    with tab_sim:
        render_simulate_tab()
    with tab_summary:
        render_summary_tab()
    with tab_results:
        render_results_tab()
