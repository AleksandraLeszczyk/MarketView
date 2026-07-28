"""SimLab Streamlit UI: agents / datasets / simulate / results.

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

from agent_stonks import clock
from agent_stonks import observability as obs
from agent_stonks.agent import AGENT_PERSONALITIES, PERSONALITY_TOOLS, _dispatch_tool
from agent_stonks.config import PALETTE
from agent_stonks.llm import DEFAULT_AGENT_MODELS, ENV_KEYS, PROVIDERS, models_for
from agent_stonks.market_hours import MARKET_TZ

from . import data as sim_data
from . import experiments as sim_experiments
from . import prompts as sim_prompts
from . import results as sim_results
from .engine import SimulationConfig, SimulationEngine
from .market import SimMarket
from .patches import simulation_context

AVATAR_DIR = Path(__file__).resolve().parent.parent / "data" / "avatars"

# submit_decision / set_tactics mutate the ledger and need a full cycle around
# them -- the hand-tester exposes only the read/analysis tools.
_UNTESTABLE_TOOLS = {"submit_decision", "set_tactics", "stand_down"}


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
        market = SimMarket(ds.symbols, [day])
        config = SimulationConfig(
            personality=personality, provider="openai", model="-", api_key="",
            symbols=ds.symbols, days=[day],
        )
        engine = SimulationEngine(market, config)
        with simulation_context(market):
            engine.seed_until(at)
            clock.set_simulated(at)
            result = _dispatch_tool(tool["name"], args, engine.app, engine.tracker)
        st.caption(f"`{tool['name']}` at {at.astimezone(MARKET_TZ).strftime('%Y-%m-%d %H:%M ET')}")
        st.json(result)


def render_agents_tab() -> None:
    keys = list(AGENT_PERSONALITIES)
    personality = st.session_state.get("agents_selected", keys[0])
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        meta = AGENT_PERSONALITIES[key]
        with col, st.container(border=True):
            avatar = AVATAR_DIR / meta.get("avatar", "")
            if avatar.exists():
                st.image(str(avatar), width=72)
            st.caption(meta["label"])
            if st.button(
                "Selected" if key == personality else "Open",
                key=f"agent_pick_{key}",
                type="primary" if key == personality else "secondary",
            ):
                st.session_state["agents_selected"] = key
                st.rerun()

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
        "Datasets are named bundles of symbols + a date range. Minute bars (04:00–20:00 ET), "
        "daily history, news, and SPY/VIX context are stored locally, deduplicated per "
        "(symbol, day) — overlapping datasets never re-download a day."
    )
    with st.form("dataset_form"):
        name = st.text_input("Dataset name", placeholder="e.g. nvda-earnings-week")
        symbols_raw = st.text_input("Symbols (comma-separated)", placeholder="NVDA, AAPL")
        col_start, col_end, col_feed = st.columns(3)
        start = col_start.date_input("Start", value=date.today() - timedelta(days=7))
        end = col_end.date_input("End", value=date.today() - timedelta(days=1))
        feed = col_feed.selectbox("Feed", ["iex", "sip"])
        col_key, col_secret = st.columns(2)
        api_key = col_key.text_input(
            "Alpaca API key", value=os.getenv("ALPACA_API_KEY", ""), type="password"
        )
        api_secret = col_secret.text_input(
            "Alpaca secret", value=os.getenv("ALPACA_SECRET", ""), type="password"
        )
        submitted = st.form_submit_button("Download dataset", icon=":material/download:", type="primary")

    if submitted:
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        if not name.strip() or not symbols:
            st.error("A dataset needs a name and at least one symbol.")
        elif not api_key or not api_secret:
            st.error("Alpaca credentials are required to download data.")
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
                f"· {len(ds.days)} trading day(s)"
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
    cols[5].metric("LLM cycles", summary.get("cycles_run", record.get("cycles_run", 0)))

    equity = record.get("equity") or []
    if equity:
        st.plotly_chart(_equity_chart(equity, record.get("starting_cash", 0.0)))

    symbols = config.get("symbols") or []
    days = [date.fromisoformat(d) for d in config.get("days") or []]
    decisions = record.get("decisions") or []
    if symbols and days:
        try:
            market = SimMarket(symbols, days)
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
    agent = AGENT_PERSONALITIES.get(personality, {}).get("label", personality)
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
            "agent": AGENT_PERSONALITIES.get(personality, {}).get("label", personality),
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
        f"**{AGENT_PERSONALITIES.get(personality, {}).get('label', personality)}** · "
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
            st.markdown(f"**{ds.name}** — {ds.start} → {ds.end}")
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
        "Agents", list(AGENT_PERSONALITIES),
        default=list(AGENT_PERSONALITIES)[:1],
        format_func=lambda k: AGENT_PERSONALITIES[k]["label"],
        key="sim_agents",
    )
    model_choices, api_keys = _render_model_picker()

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
        # Judging with the agent's own model is per-combination; a single
        # explicit judge is shared by every experiment in the grid.
        judge_override = run_judge and st.checkbox(
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

    combos = [
        (personality, provider, model, name)
        for name in selected_names
        for personality in personalities
        for provider, model in model_choices
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
        labels = ", ".join(AGENT_PERSONALITIES[p]["label"] for p in overridden)
        st.caption(f":material/edit: Runs with a **modified** prompt (Agents tab): {labels}.")
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
            st.caption(
                f"{len(personalities)} agent(s) × {len(model_choices)} model(s) × "
                f"{len(selected_names)} dataset(s) = {len(combos)} combination(s)"
                + (f", {len(combos) - len(to_queue)} skipped" if skip_tested else "")
            )
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
        elif missing_keys:
            st.error(f"An API key is required for: {', '.join(missing_keys)}.")
        elif run_judge and judge_override and not judge_api_key:
            st.error(f"An API key for the judge provider ({judge_provider}) is required.")
        else:
            for personality, provider, model, name in to_queue:
                scope = dataset_scope[name]
                sim_experiments.submit(name, {
                    "personality": personality,
                    "provider": provider,
                    "model": model,
                    "api_key": api_keys[provider],
                    "symbols": scope["symbols"],
                    "days": scope["days"],
                    "starting_cash": float(starting_cash),
                    "cycle_minutes": int(cycle_minutes),
                    "max_cycles_per_day": int(max_cycles),
                    "system_prompt_override": sim_prompts.get_override(personality),
                    "run_judge": bool(run_judge),
                    "judge_provider": judge_provider or provider,
                    "judge_model": judge_model or model,
                    "judge_api_key": judge_api_key or api_keys[provider],
                })
            sim_experiments.tick(int(st.session_state.get(MAX_PARALLEL_KEY, 2)))
            st.toast(f"{len(to_queue)} experiment(s) queued",
                     icon=":material/rocket_launch:")
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 4 — results
# ---------------------------------------------------------------------------

_BREAKDOWN_DIMENSIONS = {"LLM model": "model", "Dataset": "dataset", "Agent": "agent"}


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
        "Agent: " + AGENT_PERSONALITIES.get(best.get("personality"), {}).get(
            "label", best.get("personality") or "?"),
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


def render_results_tab() -> None:
    runs = _runs()
    if not runs:
        st.info("No stored runs yet — queue an experiment in the Simulate tab.")
        return

    st.markdown("##### Breakdown")
    dim_label = st.segmented_control(
        "Break down by", list(_BREAKDOWN_DIMENSIONS), default="LLM model",
        key="results_breakdown_dim",
    ) or "LLM model"
    dimension = _BREAKDOWN_DIMENSIONS[dim_label]
    rows = sim_results.breakdown(runs, dimension)
    if dimension == "agent":
        for row in rows:
            row["group"] = AGENT_PERSONALITIES.get(row["group"], {}).get(
                "label", row["group"])
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

    st.divider()
    st.markdown("##### Runs")
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
    if st.button("Delete this run", icon=":material/delete:"):
        sim_results.delete_run(selected)
        st.session_state.pop("last_run_id", None)
        st.rerun()


# ---------------------------------------------------------------------------

def build_ui() -> None:
    st.set_page_config(page_title="AgentStonks SimLab", page_icon="🧪", layout="wide")
    st.title("SimLab — strategy testing")
    st.caption(
        "Replay the trading agents against stored historical sessions: same prompts, same "
        "tools, same execution path as live — hours of tape in minutes of simulation."
    )
    tab_agents, tab_datasets, tab_sim, tab_results = st.tabs(
        [":material/smart_toy: Agents", ":material/database: Datasets",
         ":material/play_circle: Simulate", ":material/insights: Results"]
    )
    with tab_agents:
        render_agents_tab()
    with tab_datasets:
        render_datasets_tab()
    with tab_sim:
        render_simulate_tab()
    with tab_results:
        render_results_tab()
