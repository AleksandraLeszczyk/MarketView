"""Experiment pipeline: queued simulations, each run in its own worker process.

An *experiment* is one queued simulation -- the full ``SimulationConfig``
payload plus lifecycle state (waiting -> running -> finished/failed) -- stored
as one JSON file under ``data/simlab/experiments/`` so both the queue and the
history of past experiments survive restarts.

Parallelism is process-based by necessity, not preference: the simulation
clock (``agent_stonks.clock``) is a module global and ``simulation_context``
swaps module attributes, so one process can host at most one running
simulation. Each experiment therefore runs as ``python -m simlab.runner <id>``
in its own subprocess, with engine/judge progress lines going to a sidecar
``<id>.log``.

``tick()`` is the whole scheduler: the UI calls it on every refresh; it reaps
workers that died without finalizing and launches waiting experiments up to
the parallelism limit. The stored config carries the API key so a queued
experiment can outlive the browser session; the key is scrubbed from the
record the moment the experiment finalizes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "simlab" / "experiments"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

WAITING = "waiting"
RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"
ACTIVE_STATUSES = (WAITING, RUNNING)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(experiment_id: str) -> Path:
    return EXPERIMENTS_DIR / f"{experiment_id}.json"


def log_path(experiment_id: str) -> Path:
    return EXPERIMENTS_DIR / f"{experiment_id}.log"


def _write(record: dict) -> None:
    """Write via temp + rename so a reader never sees a half-written record
    (the worker process and the UI both touch these files)."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(record["experiment_id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def submit(dataset: str, config: dict) -> dict:
    """Queue one experiment. `config` is the JSON-ready SimulationConfig
    payload (days as ISO strings) plus ``run_judge``."""
    experiment_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    record = {
        "experiment_id": experiment_id,
        "created_at": _now_iso(),
        "status": WAITING,
        "dataset": dataset,
        "config": config,
        "pid": None,
        "started_at": None,
        "finished_at": None,
        "run_id": None,
        "error": None,
        "result_summary": None,
    }
    _write(record)
    return record


def get_experiment(experiment_id: str) -> Optional[dict]:
    try:
        return json.loads(_path(experiment_id).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_experiments() -> list[dict]:
    """All experiment records, newest first (ids sort chronologically)."""
    if not EXPERIMENTS_DIR.exists():
        return []
    records = []
    for path in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def has_active() -> bool:
    return any(e["status"] in ACTIVE_STATUSES for e in list_experiments())


def update(experiment_id: str, **fields) -> Optional[dict]:
    record = get_experiment(experiment_id)
    if record is None:
        return None
    record.update(fields)
    _write(record)
    return record


def finalize(
    experiment_id: str,
    status: str,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
    result_summary: Optional[dict] = None,
) -> Optional[dict]:
    """Terminal transition (finished/failed): stamp the end time and scrub the
    API key -- it was only stored so the worker could outlive the browser."""
    record = get_experiment(experiment_id)
    if record is None:
        return None
    record.update(
        status=status,
        run_id=run_id,
        error=error,
        result_summary=result_summary,
        finished_at=_now_iso(),
    )
    record["config"] = {
        **(record.get("config") or {}), "api_key": "", "judge_api_key": "",
    }
    _write(record)
    return record


def delete_experiment(experiment_id: str) -> None:
    for path in (_path(experiment_id), log_path(experiment_id)):
        if path.exists():
            path.unlink()


def clear_finished() -> int:
    """Drop finished/failed experiment records (run records stay untouched)."""
    cleared = 0
    for exp in list_experiments():
        if exp["status"] in (FINISHED, FAILED):
            delete_experiment(exp["experiment_id"])
            cleared += 1
    return cleared


def last_log_line(experiment_id: str) -> str:
    try:
        lines = log_path(experiment_id).read_text().strip().splitlines()
        return lines[-1] if lines else ""
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _launch(experiment_id: str):
    """Start the worker process; stdout/stderr stream to the sidecar log."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(log_path(experiment_id), "ab")
    return subprocess.Popen(
        [sys.executable, "-m", "simlab.runner", experiment_id],
        cwd=_PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def spawn(experiment_id: str) -> None:
    proc = _launch(experiment_id)
    update(experiment_id, status=RUNNING, pid=proc.pid, started_at=_now_iso())


def tick(max_parallel: int) -> None:
    """One scheduler pass: reap dead workers, then fill free slots (oldest
    waiting first). Called from the UI on every pipeline refresh."""
    experiments = list_experiments()
    running = []
    for exp in experiments:
        if exp["status"] != RUNNING:
            continue
        if _pid_alive(exp.get("pid")):
            running.append(exp)
        else:
            finalize(exp["experiment_id"], FAILED, error="worker process died")
    waiting = sorted(
        (e for e in experiments if e["status"] == WAITING),
        key=lambda e: (e.get("created_at", ""), e["experiment_id"]),
    )
    for exp in waiting[: max(0, int(max_parallel) - len(running))]:
        spawn(exp["experiment_id"])
