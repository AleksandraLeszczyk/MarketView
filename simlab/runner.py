"""Worker process for one queued experiment: ``python -m simlab.runner <id>``.

Spawned by ``simlab.experiments.tick()``. Owns the whole process, which is what
makes parallel experiments safe: the pinned simulation clock and the
``simulation_context`` module patches never escape this worker. Loads the
experiment record, replays it through the real engine, optionally judges the
run, persists the run record (``simlab.results``), and finalizes the
experiment file that the UI polls. Progress lines go to stdout, which the
spawner redirects to the experiment's sidecar log.
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Load the project-root .env by explicit path: find_dotenv()'s stack-walking
# does not reliably resolve it under ``python -m`` in a detached session, and
# a worker without the API keys dies on its first LLM call.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

from simlab import experiments  # noqa: E402
from simlab import judge as sim_judge  # noqa: E402
from simlab import prompts as sim_prompts  # noqa: E402
from simlab import results as sim_results  # noqa: E402
from simlab.engine import SimulationConfig, SimulationEngine  # noqa: E402
from simlab.market import SimMarket  # noqa: E402


def _progress(msg: str) -> None:
    print(msg, flush=True)


def run_experiment(experiment_id: str) -> dict:
    exp = experiments.get_experiment(experiment_id)
    if exp is None:
        raise RuntimeError(f"unknown experiment {experiment_id}")
    cfg = exp["config"]
    days = [date.fromisoformat(d) for d in cfg["days"]]
    config = SimulationConfig(
        personality=cfg["personality"],
        provider=cfg["provider"],
        model=cfg["model"],
        api_key=cfg["api_key"],
        symbols=cfg["symbols"],
        days=days,
        starting_cash=float(cfg["starting_cash"]),
        cycle_minutes=int(cfg["cycle_minutes"]),
        max_cycles_per_day=int(cfg["max_cycles_per_day"]),
        system_prompt_override=cfg.get("system_prompt_override"),
    )
    market = SimMarket(config.symbols, days)
    engine = SimulationEngine(market, config, progress=_progress)
    result = engine.run()
    _progress(
        f"Replay finished: {result.cycles_run} LLM cycle(s), "
        f"{len(result.decisions)} ledger entrie(s)."
    )
    summary = sim_results.summarize_run(result, market)
    judge_report = None
    if cfg.get("run_judge"):
        strategy_prompt = cfg.get("system_prompt_override") or sim_prompts.get_prompt(
            config.personality
        )
        # The judge LLM is configurable and falls back to the agent's own
        # provider/model, so older experiment records still judge.
        judge_report = sim_judge.judge_run(
            result.decisions, summary, strategy_prompt, market,
            cfg.get("judge_provider") or config.provider,
            cfg.get("judge_api_key") or config.api_key,
            cfg.get("judge_model") or config.model,
            progress=_progress,
        )
    return sim_results.save_run(
        result, summary, judge_report, dataset_name=exp.get("dataset", "")
    )


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("usage: python -m simlab.runner <experiment_id>")
    experiment_id = argv[1]
    try:
        record = run_experiment(experiment_id)
    except Exception as exc:
        experiments.finalize(experiment_id, experiments.FAILED, error=str(exc))
        raise
    summary = record.get("summary") or {}
    judge = record.get("judge") or {}
    experiments.finalize(
        experiment_id,
        experiments.FINISHED,
        run_id=record["run_id"],
        error=record.get("error"),
        result_summary={
            "return_pct": summary.get("return_pct"),
            "profit_efficiency": summary.get("profit_efficiency"),
            "judge_overall": judge.get("overall_score"),
            "trades_filled": summary.get("trades_filled"),
        },
    )
    _progress(f"Experiment {experiment_id} finished (run {record['run_id']}).")


if __name__ == "__main__":
    main(sys.argv)
