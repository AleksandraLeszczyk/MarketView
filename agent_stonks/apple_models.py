"""The models Apple Trader can run on, behind one interface.

Apple Trader's rules are fixed -- buy a momentum-regime change into positive
that the model expects to hold, sell on a trailing stop -- but *which* model
answers the first half is a choice, and TimeToChange2 has produced more than
one answer to it. This module is where that choice lives, so the trader, the
loop, SimLab and the UI ask for "the model named X" and never branch on which
one they got.

The two on offer
----------------
`persistence`  the incumbent: a gradient-boosted classifier trained on the
               label directly (notebook 04). Out-of-fold AUC 0.82 over all
               regime changes, **0.50** over the ones that already pass the
               observable `pre_dwell >= 15` pre-condition. A filter that
               separates the impossible from the possible.
`nbeats`       a five-seed N-BEATS ensemble that forecasts the next 15 bars of
               momentum, replays the regime trigger over 500 sampled futures
               and reports the fraction that survive (notebooks 06-07). 0.91
               full-label AUC and **0.67 +/- 0.07** on that same hard half --
               the only entrant above chance on all four walk-forward folds.

The second number is the only one worth choosing on, and the gap it describes
is real but small: 35 events, with a bootstrap interval that only just excludes
chance. Notebook 07 replayed one held-out session through both and they made
*identical* trades -- a day with four candidates cannot resolve a difference
measured over 35. So this is offered as a switch, not as a promotion.

What "one interface" means concretely
-------------------------------------
Both loaders return a bundle with `feature_columns`, `seq_len`, `threshold`,
`settings` and `metrics`, and either a `pipeline` or a `score` --
`persistence_model.predict_proba` accepts both. Everything before the scoring
step (bars, sessions, momentum, regimes, all 25 features, the 20-bar window)
comes from `persistence_model` either way, so the two models are handed the
same array on the same bars and differ only in what they do with it.

Both are optional in the same way the rest of the ML surface is: a missing file
or a missing dependency makes a model *unavailable*, reported as such, rather
than an agent that silently never trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import persistence_model
from .config import APPLE_TRADER_MODEL


@dataclass(frozen=True)
class AppleModel:
    """One model Apple Trader can be pointed at."""

    key: str
    label: str
    # One line for a picker: what the model is, not how well it scores.
    summary: str
    # What has to be installed for `load` to be able to return anything.
    requires: str
    # Whether it can be asked about a change that has not happened yet, which
    # is what Apple Trader's `anticipate` entry needs. Only a forecaster can:
    # see `persistence_model.anticipates`, which is the authority at run time
    # (it reads the loaded bundle). This flag is the same fact available before
    # a 200 MB dependency is imported, so a picker can label the choice.
    anticipates: bool
    # Registry data, not the entry point: everything loads through `load(key)`
    # below, so there is one seam for tests to replace and one place a caller
    # can reach a model from.
    load: Callable[[], "dict | None"]
    path: Callable[[], Path]


def _load_persistence() -> "dict | None":
    """The incumbent bundle.

    A wrapper rather than `persistence_model.load_bundle` itself, so the lookup
    happens when the model is asked for. Binding the function object into the
    registry at import time would freeze it past any later replacement -- which
    is exactly what a test that stubs out a missing model does.
    """
    return persistence_model.load_bundle()


def _persistence_path() -> Path:
    return persistence_model.model_path()


def _load_nbeats() -> "dict | None":
    """The N-BEATS bundle, or None if torch is not installed.

    Imported here rather than at module scope so that `import apple_models`
    stays free of PyTorch: the incumbent model needs none of it, and the
    default configuration must not require a 200 MB dependency to start.
    """
    try:
        from . import nbeats_model
    except ImportError:
        return None
    return nbeats_model.load_bundle()


def _nbeats_path() -> Path:
    try:
        from . import nbeats_model
    except ImportError:
        return Path("timetochange2_nbeats.pt")
    return nbeats_model.model_path()


PERSISTENCE_KEY = "persistence"
NBEATS_KEY = "nbeats"

MODELS: "dict[str, AppleModel]" = {
    PERSISTENCE_KEY: AppleModel(
        key=PERSISTENCE_KEY,
        label="Persistence classifier (HGB)",
        summary=(
            "Gradient-boosted classifier trained on the persistence label directly "
            "(TimeToChange2 notebook 04). Strong at rejecting changes that cannot "
            "hold, no better than a coin flip at ranking the ones that can."
        ),
        requires="scikit-learn and joblib",
        anticipates=False,
        load=_load_persistence,
        path=_persistence_path,
    ),
    NBEATS_KEY: AppleModel(
        key=NBEATS_KEY,
        label="N-BEATS forecast → persistence",
        summary=(
            "Five seeds of N-BEATS forecast the next 15 bars of momentum; 500 "
            "sampled futures are run through the real regime trigger and the "
            "fraction that survive is the probability (notebooks 06-07). The only "
            "entrant above chance on all four walk-forward folds of the hard half."
        ),
        requires="PyTorch, plus the residual sidecar beside the checkpoint",
        anticipates=True,
        load=_load_nbeats,
        path=_nbeats_path,
    ),
}

DEFAULT_MODEL = APPLE_TRADER_MODEL if APPLE_TRADER_MODEL in MODELS else PERSISTENCE_KEY


def keys() -> "list[str]":
    """Every model key, in the order a picker should offer them."""
    return list(MODELS)


def get(key: "str | None") -> AppleModel:
    """The named model, falling back to the default for an unknown key.

    Unknown keys reach here from experiment records written before a model
    existed or after one was renamed; a stored run should still replay on the
    default rather than crash the Results page.
    """
    return MODELS.get(key or DEFAULT_MODEL) or MODELS[DEFAULT_MODEL]


def load(key: "str | None") -> "dict | None":
    """The named model's bundle, or None when it cannot be assembled."""
    return get(key).load()


def unavailable_reason(key: "str | None") -> str:
    """Why `load` returned None, in the terms a user can act on."""
    model = get(key)
    return (
        f"No {model.label} model at {model.path()} "
        f"(or {model.requires} are not installed)."
    )


def threshold(key: "str | None", bundle: "dict | None" = None) -> float:
    """The cut-off the named model chose on its own validation block.

    Not comparable across models: the classifier's is a posterior and N-BEATS'
    is a survival probability times a hard gate, so 0.07 and 0.05 are the same
    kind of number only by coincidence.
    """
    return persistence_model.model_threshold(
        bundle if bundle is not None else load(key)
    )
