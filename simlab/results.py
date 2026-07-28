"""Simulation run records: scoring, persistence, and optional Langfuse export.

A finished simulation is summarized (profit, trade counts, an oracle
best-round-trip ceiling and the profit efficiency against it -- the same
oracle the live daily scoring uses), optionally judged by an LLM
(``simlab.judge``), and saved as one JSON file under ``data/simlab/runs/``.

When Langfuse is configured the headline metrics are also registered there as
scores on a ``simlab-run`` trace, so runs line up next to the live agent's
traces and daily scores in the same project. The local JSON stays the source
of truth -- Langfuse is an export target, never a dependency.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_stonks import observability as obs

from .engine import SimulationResult
from .market import SimMarket

RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "simlab" / "runs"

# Stand-in group/filter key for runs saved without a dataset name.
NO_DATASET = "(no dataset)"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def oracle_best_round_trip(closes: list[float]) -> float:
    """Max single-round-trip profit percentage over an ordered close series
    (buy at the running minimum, sell at the best later price). 0 when the
    series never rises."""
    best = 0.0
    low: Optional[float] = None
    for price in closes:
        if price <= 0:
            continue
        if low is None or price < low:
            low = price
            continue
        best = max(best, (price / low - 1.0) * 100.0)
    return best


def summarize_run(result: SimulationResult, market: SimMarket) -> dict:
    """Headline stats: return, fees, trade counts, and the oracle ceiling."""
    decisions = result.decisions
    fills = [d for d in decisions if d.get("status") == "filled" and d.get("action") in ("buy", "sell")]
    per_symbol_oracle = {}
    for symbol, series in market.series.items():
        closes = [float(b["c"]) for b in series.minute_bars if b.get("c") is not None]
        per_symbol_oracle[symbol] = round(oracle_best_round_trip(closes), 3)
    oracle_ceiling = max(per_symbol_oracle.values(), default=0.0)
    return_pct = (
        (result.final_value / result.starting_cash - 1.0) * 100.0 if result.starting_cash else 0.0
    )
    efficiency = (return_pct / oracle_ceiling) if oracle_ceiling > 0 else None
    return {
        "starting_cash": result.starting_cash,
        "final_value": round(result.final_value, 2),
        "profit": round(result.final_value - result.starting_cash, 2),
        "return_pct": round(return_pct, 4),
        "total_fees": round(sum(d.get("fee") or 0.0 for d in decisions), 2),
        "cycles_run": result.cycles_run,
        "trades_filled": len(fills),
        "buys": sum(1 for d in fills if d["action"] == "buy"),
        "sells": sum(1 for d in fills if d["action"] == "sell"),
        "tactics_armed": sum(1 for d in decisions if d.get("action") == "tactics"),
        "alerts_set": sum(1 for d in decisions if d.get("action") == "alert"),
        "oracle_best_round_trip_pct": per_symbol_oracle,
        "oracle_ceiling_pct": round(oracle_ceiling, 3),
        "profit_efficiency": round(efficiency, 4) if efficiency is not None else None,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_run(
    result: SimulationResult,
    summary: dict,
    judge_report: "dict | None" = None,
    dataset_name: str = "",
) -> dict:
    """Persist one finished run; returns the full stored record."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        **asdict(result),
        "summary": summary,
        "judge": judge_report,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2, default=str))
    _export_to_langfuse(record)
    return record


def list_runs() -> list[dict]:
    """Stored run records, newest first (full records -- they're small)."""
    if not RUNS_DIR.exists():
        return []
    records = []
    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def cycle_error_count(record: dict) -> int:
    """Cycle-level failures logged during a run -- in practice almost always a
    failed LLM call (rate limit, rejected key, refused request). The engine
    logs one and carries on, so a run can finish and still be a degraded test."""
    return sum(
        1 for entry in (record.get("agent_log") or []) if entry.get("type") == "error"
    )


def find_prior_runs(
    runs: list[dict], personality: str, provider: str, model: str, dataset: str
) -> dict[str, list[dict]]:
    """Stored runs for this exact (agent, provider/model, dataset) combination,
    newest first, split into ``clean`` -- finished end to end with no failed
    cycles, so re-running buys nothing -- and ``degraded``, which are worth
    running again."""
    clean, degraded = [], []
    for record in runs:
        config = record.get("config_summary") or {}
        if (
            config.get("personality") != personality
            or config.get("provider") != provider
            or config.get("model") != model
            or (record.get("dataset") or "") != dataset
        ):
            continue
        healthy = (
            not record.get("error")
            and not record.get("interrupted")
            and (record.get("cycles_run") or 0) > 0
            and cycle_error_count(record) == 0
        )
        (clean if healthy else degraded).append(record)
    return {"clean": clean, "degraded": degraded}


def store_signature() -> tuple:
    """Cheap fingerprint of the run store (file name + mtime). Lets the UI cache
    parsed run records and drop them the moment a run is added or deleted."""
    if not RUNS_DIR.exists():
        return ()
    try:
        return tuple(sorted((p.name, p.stat().st_mtime) for p in RUNS_DIR.glob("*.json")))
    except OSError:
        return ()


def model_key(record: dict) -> str:
    """``provider/model`` identity of a stored run -- the same string the model
    breakdown groups on, so filters and rows always agree."""
    config = record.get("config_summary") or {}
    return f"{config.get('provider', '?')}/{config.get('model', '?')}"


def dataset_key(record: dict) -> str:
    return record.get("dataset") or NO_DATASET


def filter_options(runs: list[dict]) -> dict[str, list[str]]:
    """The datasets and models actually present in the stored runs, sorted --
    the option lists for the Results filters."""
    return {
        # Named datasets first, the "(no dataset)" catch-all last.
        "datasets": sorted({dataset_key(r) for r in runs},
                           key=lambda name: (name == NO_DATASET, name)),
        "models": sorted({model_key(r) for r in runs}),
    }


def filter_runs(
    runs: list[dict],
    datasets: "list[str] | None" = None,
    models: "list[str] | None" = None,
) -> list[dict]:
    """Runs matching every non-empty filter. An empty (or omitted) filter means
    "no restriction on this dimension", so no selection shows everything."""
    dataset_set, model_set = set(datasets or ()), set(models or ())
    return [
        record
        for record in runs
        if (not dataset_set or dataset_key(record) in dataset_set)
        and (not model_set or model_key(record) in model_set)
    ]


def breakdown(runs: list[dict], by: str) -> list[dict]:
    """Aggregate stored runs along one dimension: ``model`` (provider/model),
    ``dataset``, or ``agent`` (personality). Averages skip runs where a metric
    is unavailable (no judge report, oracle ceiling of 0). Each row also carries
    ``best_run`` -- the identity (model, agent, dataset, run id) of the single
    run behind ``best_return_pct``, since the other two dimensions are invisible
    in a breakdown along the third."""
    if by not in ("model", "dataset", "agent"):
        raise ValueError(f"unknown breakdown dimension: {by}")
    groups: dict[str, dict] = {}
    for record in runs:
        config = record.get("config_summary") or {}
        summary = record.get("summary") or {}
        if by == "model":
            key = model_key(record)
        elif by == "dataset":
            key = dataset_key(record)
        else:
            key = config.get("personality") or "?"
        group = groups.setdefault(
            key, {"count": 0, "returns": [], "efficiencies": [], "scores": [], "best": None}
        )
        group["count"] += 1
        if summary.get("return_pct") is not None:
            return_pct = float(summary["return_pct"])
            group["returns"].append(return_pct)
            if group["best"] is None or return_pct > group["best"]["return_pct"]:
                group["best"] = {
                    "return_pct": return_pct,
                    "run_id": record.get("run_id") or "",
                    "provider": config.get("provider") or "",
                    "model": config.get("model") or "",
                    "personality": config.get("personality") or "",
                    "dataset": record.get("dataset") or "",
                }
        if summary.get("profit_efficiency") is not None:
            group["efficiencies"].append(float(summary["profit_efficiency"]))
        judge = record.get("judge") or {}
        if judge.get("overall_score") is not None:
            group["scores"].append(float(judge["overall_score"]))

    def _avg(values: list[float]) -> Optional[float]:
        return round(sum(values) / len(values), 4) if values else None

    rows = [
        {
            "group": key,
            "runs": g["count"],
            "avg_return_pct": _avg(g["returns"]),
            "best_return_pct": round(max(g["returns"]), 4) if g["returns"] else None,
            "best_run": g["best"],
            "avg_profit_efficiency": _avg(g["efficiencies"]),
            "avg_judge_score": _avg(g["scores"]),
        }
        for key, g in groups.items()
    ]
    rows.sort(key=lambda r: (r["avg_profit_efficiency"] is None,
                             -(r["avg_profit_efficiency"] or 0.0)))
    return rows


def delete_run(run_id: str) -> None:
    path = RUNS_DIR / f"{run_id}.json"
    if path.exists():
        path.unlink()


def delete_all_runs() -> int:
    """Drop every stored run; returns how many records were removed. Only the
    local store is touched -- scores already exported to Langfuse stay there."""
    if not RUNS_DIR.exists():
        return 0
    removed = 0
    for path in RUNS_DIR.glob("*.json"):
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Langfuse export (no-op when unconfigured, like the rest of observability)
# ---------------------------------------------------------------------------

def _export_to_langfuse(record: dict) -> None:
    if not obs.is_enabled():
        return
    summary = record.get("summary") or {}
    config = record.get("config_summary") or {}
    label = (
        f"simlab-run:{config.get('personality')}:{record.get('dataset') or ','.join(config.get('symbols', []))}"
    )
    comment = json.dumps(
        {"run_id": record["run_id"], **{k: config.get(k) for k in ("personality", "model", "days")}}
    )
    scores = {
        "sim-return-pct": summary.get("return_pct"),
        "sim-profit-efficiency": summary.get("profit_efficiency"),
    }
    judge = record.get("judge") or {}
    if judge.get("overall_score") is not None:
        scores["sim-judge-overall"] = judge["overall_score"]
    for name, value in scores.items():
        if value is None:
            continue
        obs.record_score(
            trace_name=label, name=name, value=float(value), comment=comment, input=summary
        )
