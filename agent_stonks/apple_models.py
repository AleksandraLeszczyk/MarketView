"""The models Apple Trader can run on, behind one interface.

*Which* saved model the agent trades on is a choice, and the notebooks have
produced more than one answer. This module is where that choice lives, so the
trader, the loop, SimLab and the UI ask for "the model named X" and never
branch on which one they got.

The three on offer
------------------
`persistence`  the incumbent: a gradient-boosted classifier trained on the
               momentum-persistence label directly (TimeToChange2 notebook 04).
               Out-of-fold AUC 0.82 over all regime changes, **0.50** over the
               ones that already pass the observable `pre_dwell >= 15`
               pre-condition. A filter that separates the impossible from the
               possible.
`nbeats`       a five-seed N-BEATS ensemble that forecasts the next 15 bars of
               momentum, replays the regime trigger over 500 sampled futures
               and reports the fraction that survive (TimeToChange2 notebooks
               06-07). 0.91 full-label AUC and **0.67 +/- 0.07** on that same
               hard half -- the only entrant above chance on all four
               walk-forward folds.
`dayrange`     a different question entirely: TimeToChange3's blend of
               LightGBM, N-BEATS and N-HiTS, plus an opening ridge, forecasting
               where the *whole session's* high and low will land, once, from
               the first five minutes. 30% of a 14-day rolling baseline's error
               removed over 129 test sessions.

The first two are graded against each other on the same 35 events, and the gap
is real but small -- notebook 07 replayed one held-out session through both and
they made *identical* trades. So `nbeats` is a switch, not a promotion.

`dayrange` is not on that scale at all and cannot be compared to them by
number. It answers a different question, on a different horizon, and it brings
its own trading rules with it.

What "one interface" means, and where it stops
----------------------------------------------
`persistence` and `nbeats` share everything before the scoring step: bars,
sessions, momentum, regimes, all 25 features and the 20-bar window come from
`persistence_model` either way, so the two are handed the same array on the
same bars and differ only in what they do with it. Both bundles carry
`feature_columns`, `seq_len`, `threshold`, `settings` and `metrics`. That is
what `strategy == STRATEGY_MOMENTUM` means.

`dayrange` shares none of it. A forecast of the day's high is not a probability
and there is no threshold to compare it against; the rules built on it are
resting price levels rather than a per-bar signal. Pretending otherwise -- one
`read_latest`, one probability, one trailing stop -- would have meant a
`turn_proba` that is really a price and a threshold that means nothing.
So the seam is drawn at the strategy instead: `AppleModel.strategy` names which
rule set a model drives, `apple_trader.build_trader` turns that into the right
state machine, and everything downstream of it (the config record, the Results
signature, both UIs) branches on the strategy rather than on the model key.

Every model is optional in the same way the rest of the ML surface is: a
missing file or a missing dependency makes it *unavailable*, reported as such,
rather than an agent that silently never trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import persistence_model
from .config import APPLE_TRADER_MODEL


# The rule sets a model can drive. A model does not merely answer a question --
# it decides which question is worth asking, and the two here are not variants
# of one another (see the module docstring).
STRATEGY_MOMENTUM = "momentum"
STRATEGY_DAYRANGE = "dayrange"


@dataclass(frozen=True)
class AppleModel:
    """One model Apple Trader can be pointed at."""

    key: str
    label: str
    # One line for a picker: what the model is, not how well it scores.
    summary: str
    # What has to be installed for `load` to be able to return anything.
    requires: str
    # Which rule set this model drives -- `apple_trader.build_trader` turns it
    # into a state machine, and every caller that needs to know which knobs
    # apply reads this rather than the key.
    strategy: str
    # Whether it can be asked about a change that has not happened yet, which
    # is what Apple Trader's `anticipate` entry needs. Only a momentum
    # forecaster can: see `persistence_model.anticipates`, which is the
    # authority at run time (it reads the loaded bundle). This flag is the same
    # fact available before a 200 MB dependency is imported, so a picker can
    # label the choice. Meaningless outside `STRATEGY_MOMENTUM`.
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


def _load_dayrange() -> "dict | None":
    """The TimeToChange3 bundle, or None if torch/LightGBM are not installed.

    Imported here rather than at module scope for the same reason `nbeats` is,
    and one more: this module pulls LightGBM in ahead of torch on purpose, and
    doing that at `import apple_models` time would impose the ordering on every
    process that only wanted to list the model names.
    """
    try:
        from . import dayrange_model
    except ImportError:
        return None
    return dayrange_model.load_bundle()


def _dayrange_path() -> Path:
    try:
        from . import dayrange_model
    except ImportError:
        return Path("timetochange3_dayrange_AAPL.joblib")
    return dayrange_model.model_path()


PERSISTENCE_KEY = "persistence"
NBEATS_KEY = "nbeats"
DAYRANGE_KEY = "dayrange"

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
        strategy=STRATEGY_MOMENTUM,
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
        strategy=STRATEGY_MOMENTUM,
        anticipates=True,
        load=_load_nbeats,
        path=_nbeats_path,
    ),
    DAYRANGE_KEY: AppleModel(
        key=DAYRANGE_KEY,
        label="Day-range forecast (TimeToChange3)",
        summary=(
            "Forecasts where the whole session's high and low will land, once, at 9:35 "
            "— an equal blend of LightGBM, N-BEATS and N-HiTS over a year of daily "
            "history, corrected by a ridge on the first five minutes and clipped to "
            "contain the opening range. It removes 30% of a 14-day rolling baseline's "
            "error over 129 test sessions. What it predicts well is the *width* of the "
            "day, not its direction, which is why the rules around it buy well below "
            "the predicted high and sell just under it."
        ),
        requires="PyTorch, LightGBM, scikit-learn and joblib, plus both .pt checkpoints",
        strategy=STRATEGY_DAYRANGE,
        # Not applicable: nothing here forecasts a momentum regime. The entry
        # mode is not part of this strategy's configuration at all.
        anticipates=False,
        load=_load_dayrange,
        path=_dayrange_path,
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


def strategy(key: "str | None") -> str:
    """Which rule set the named model drives."""
    return get(key).strategy


def is_momentum(key: "str | None") -> bool:
    """Whether the named model drives the momentum-regime rules -- i.e. whether
    the entry mode, the probability threshold, the trailing stop and the
    reversal exit mean anything for it."""
    return get(key).strategy == STRATEGY_MOMENTUM


def threshold(key: "str | None", bundle: "dict | None" = None) -> float:
    """The cut-off the named momentum model chose on its own validation block.

    Not comparable across models: the classifier's is a posterior and N-BEATS'
    is a survival probability times a hard gate, so 0.07 and 0.05 are the same
    kind of number only by coincidence. And not defined at all outside
    `STRATEGY_MOMENTUM` -- a day-range forecast is a price, so there is no
    probability to cut. Callers should gate on `is_momentum` rather than read
    the 0.5 that a bundle without a threshold falls back to.
    """
    return persistence_model.model_threshold(
        bundle if bundle is not None else load(key)
    )
