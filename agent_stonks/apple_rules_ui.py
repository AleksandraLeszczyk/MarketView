"""The Streamlit rule builder Apple Trader 2 is configured in.

One module rather than one per app: the live dashboard and SimLab both offer
this agent, the builder is the largest control surface in either of them, and
two copies of a nested list editor would drift the first time a field was added.
`rules_panel(prefix)` is the whole public surface -- the prefix namespaces the
widget keys so both apps can render it in one Python process without colliding.

How the state works, because a list editor in Streamlit has exactly one hard
part. The rules live in `st.session_state` as plain dicts, and every rule and
condition carries an `id` from a monotonic counter. Widget keys are built from
that id rather than from the row's position, so inserting or deleting a row
cannot leave a widget holding the value that belonged to its neighbour. Loading
a preset hands out fresh ids for the same reason: the new widgets have keys
nothing has written to, which is the only way to reseed a widget Streamlit
refuses to let you assign to after it has rendered.

The instrument is picked here too, above the rules, because it decides what the
rules may say: the signal picker offers `apple_rules.signals_for(ticker)`, so a
model nothing was fitted on for that symbol is simply not in the list. Changing
the instrument does *not* rewrite the rules -- silently editing somebody's
strategy because they looked at another symbol would be worse than the problem
-- so a set carried across is left intact, the conditions that no longer read
are marked in place, and `_verdict` refuses to call it runnable until they are
gone. `presets_for` narrows the same way, and every instrument has at least the
model-free preset to start from.
"""

from __future__ import annotations

import streamlit as st

from . import apple_models, apple_rules as ar
from .apple_trader2 import AppleTrader2Config

# Value inputs step by what the number means -- a probability moves in
# hundredths and a price does not.
_STEP = {"prob": 0.01, "%": 0.05, "×ADR": 0.05, "$": 0.5, "σ": 0.1, "bars": 1.0,
         "min": 1.0, "sh": 1.0, "": 1.0}
_FORMAT = {"prob": "%.2f", "%": "%.2f", "×ADR": "%.2f", "$": "%.2f", "σ": "%.2f"}

_ACTION_LABEL = {ar.BUY: ":material/trending_up: Buy", ar.SELL: ":material/trending_down: Sell"}
_JOIN_LABEL = {ar.JOIN_ALL: "ALL of these (AND)", ar.JOIN_ANY: "ANY of these (OR)"}
_OP_LABEL = {ar.OP_ABOVE: "at or above ≥", ar.OP_BELOW: "at or below ≤"}


# --- session state -----------------------------------------------------------


def _state_key(prefix: str) -> str:
    return f"{prefix}_rules"


def _ticker_key(prefix: str) -> str:
    return f"{prefix}_ticker"


def _next_id(prefix: str) -> int:
    key = f"{prefix}_next_id"
    st.session_state[key] = st.session_state.get(key, 0) + 1
    return st.session_state[key]


def _with_ids(prefix: str, ruleset: ar.RuleSet) -> "list[dict]":
    """A rule set as the editor holds it: plain dicts, each with a fresh id."""
    rules = []
    for item in ruleset.items:
        raw = {
            "id": _next_id(prefix),
            "action": item.action,
            "size_mode": item.size_mode,
            "size": item.size,
            "join": item.join,
            "label": item.label,
            "enabled": item.enabled,
            "cooldown_bars": item.cooldown_bars,
            "conditions": [
                {"id": _next_id(prefix), "field": c.field, "op": c.op, "value": c.value}
                for c in item.conditions
            ],
        }
        rules.append(raw)
    return rules


def _rules(prefix: str, ticker: "str | None" = None) -> "list[dict]":
    key = _state_key(prefix)
    if key not in st.session_state:
        # A fresh builder starts on a preset that runs where it is pointed --
        # on a symbol with no saved model that is the model-free one.
        st.session_state[key] = _with_ids(prefix, ar.preset(None, ticker))
    return st.session_state[key]


def load_preset(prefix: str, name: str, ticker: "str | None" = None) -> None:
    """Replace the editor's contents with a named preset."""
    st.session_state[_state_key(prefix)] = _with_ids(prefix, ar.preset(name, ticker))


def set_rules(prefix: str, ruleset: ar.RuleSet) -> None:
    """Seed the editor from an existing rule set (a stored SimLab config, say)."""
    st.session_state[_state_key(prefix)] = _with_ids(prefix, ruleset)


# --- mutations, as callbacks so the rerun happens after the change ------------


def _add_rule(prefix: str) -> None:
    _rules(prefix).append(
        {
            "id": _next_id(prefix),
            "action": ar.BUY,
            "size_mode": ar.SIZE_PCT,
            "size": 95.0,
            "join": ar.JOIN_ALL,
            "label": "",
            "enabled": True,
            "cooldown_bars": 0,
            "conditions": [_new_condition(prefix)],
        }
    )


def _new_condition(prefix: str) -> dict:
    return {"id": _next_id(prefix), "field": "bar.price", "op": ar.OP_BELOW, "value": 100.0}


def _add_condition(prefix: str, index: int) -> None:
    _rules(prefix)[index]["conditions"].append(_new_condition(prefix))


def _drop_condition(prefix: str, index: int, cond_index: int) -> None:
    conditions = _rules(prefix)[index]["conditions"]
    if 0 <= cond_index < len(conditions):
        conditions.pop(cond_index)


def _drop_rule(prefix: str, index: int) -> None:
    rules = _rules(prefix)
    if 0 <= index < len(rules):
        rules.pop(index)


def _move_rule(prefix: str, index: int, delta: int) -> None:
    rules = _rules(prefix)
    target = index + delta
    if 0 <= target < len(rules):
        rules[index], rules[target] = rules[target], rules[index]


# --- the panel ---------------------------------------------------------------


def rules_panel(
    prefix: str,
    *,
    defaults: "AppleTrader2Config | None" = None,
    symbols: "list[str] | None" = None,
) -> AppleTrader2Config:
    """Render the builder and return the configuration it currently describes.

    Called on every rerun, so it both draws the widgets and reads them back: the
    editor's dicts are updated in place from the widget values, which is what
    keeps an edit alive across the rerun that an add or a delete causes.

    `symbols` is what the surrounding app can actually supply bars for -- the
    streamed tickers live, the dataset's symbols in SimLab. They are offered
    alongside the modelled ones, and the picker still accepts anything typed
    into it, because a rule set written on the tape runs on any symbol the app
    is streaming.
    """
    if defaults is not None and _state_key(prefix) not in st.session_state:
        set_rules(prefix, defaults.rules)
    flatten_default = (defaults or AppleTrader2Config()).flatten_before_close_min

    ticker = _instrument_row(prefix, defaults, symbols)
    _preset_row(prefix, ticker)
    rules = _rules(prefix, ticker)
    for index, raw in enumerate(rules):
        _rule_editor(prefix, index, raw, len(rules), ticker)

    st.button(
        "Add a rule", icon=":material/add:", key=f"{prefix}_add_rule",
        on_click=_add_rule, args=(prefix,),
    )

    flatten = st.number_input(
        "Flatten the book this many minutes before the close",
        min_value=1, max_value=60, value=int(flatten_default), step=1,
        key=f"{prefix}_flatten",
        help=(
            "Not one of the rules, and deliberately so: every signal above is intraday "
            "and none of them survives the overnight gap, so anything still open is "
            "closed before the bell whatever the list says. No new position is opened "
            "inside this window either."
        ),
    )

    config = AppleTrader2Config(
        rules=_ruleset_from(rules), flatten_before_close_min=int(flatten), ticker=ticker
    )
    _verdict(config)
    return config


def _instrument_row(
    prefix: str,
    defaults: "AppleTrader2Config | None",
    symbols: "list[str] | None",
) -> str:
    """The symbol every rule below is evaluated against.

    The options are the modelled tickers first (those are the ones where the
    full catalogue is available), then whatever the app is streaming, and the
    box accepts a symbol typed into it as well -- being unmodelled is not being
    untradeable, it just means the rules have to be written on the tape.
    """
    modelled = apple_models.tickers()
    current = st.session_state.get(_ticker_key(prefix)) or (
        defaults or AppleTrader2Config()
    ).ticker
    options = list(modelled)
    # The current value is in the list whatever it is -- including a symbol
    # typed into the box on an earlier rerun, which is otherwise in none of
    # these sources and would be dropped the next time the app reruns.
    for symbol in [*(symbols or []), current]:
        symbol = str(symbol).strip().upper()
        if symbol and symbol not in options:
            options.append(symbol)

    ticker = st.selectbox(
        "Instrument", options,
        index=options.index(current) if current in options else 0,
        key=_ticker_key(prefix), accept_new_options=True,
        help=(
            "Which symbol's minute bars the rules are read off. "
            + ", ".join(modelled)
            + " have saved models, so their forecasts appear as signals; on any other "
            "symbol the catalogue narrows to what is computed from the tape — price, "
            "the momentum regime, the position and the clock — which means the same "
            "thing everywhere. The symbol has to be one this app is streaming (or one "
            "in the simulated dataset); it is not checked here."
        ),
    )
    ticker = str(ticker or apple_models.DEFAULT_TICKER).strip().upper()

    keys = apple_models.keys_for(ticker)
    if keys:
        st.caption(
            f":material/model_training: Models fitted on {ticker}: "
            + ", ".join(apple_models.get(k).label for k in keys)
            + "."
        )
    else:
        st.caption(
            f":material/model_training: No saved model covers {ticker}, so the rules "
            "below can only read the tape, the momentum regime, the position and the "
            "clock. Everything a model would add is absent rather than approximated."
        )
    return ticker


def _preset_row(prefix: str, ticker: str) -> None:
    names = ar.presets_for(ticker)
    default = ar.default_preset(ticker)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        name = st.selectbox(
            "Start from", names, index=names.index(default) if default in names else 0,
            # Scoped to the instrument: the list of presets that can run changes
            # with it, and a widget holding a name that is no longer an option
            # would be a stale selection rather than a choice.
            key=f"{prefix}_preset_{ticker}",
            help=(
                "Only the presets that can run on this instrument are listed. The first "
                "three (on AAPL) are the strategies Apple Trader 1 ships, written out as "
                "rules — load one to see how a shipped strategy decomposes, or to have "
                "something measured to edit away from. Loading replaces everything below."
            ),
        )
        st.button(
            "Load", icon=":material/download:", key=f"{prefix}_load_preset",
            on_click=load_preset, args=(prefix, name, ticker),
        )


def _rule_editor(prefix: str, index: int, raw: dict, total: int, ticker: str) -> None:
    """One action item. Reads its widgets back into `raw` as it goes."""
    rid = raw["id"]
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            raw["action"] = (
                st.segmented_control(
                    f"Rule {index + 1}", ar.ACTIONS, default=raw["action"],
                    format_func=lambda a: _ACTION_LABEL[a], key=f"{prefix}_r{rid}_action",
                )
                or raw["action"]
            )
            raw["size"] = float(
                st.number_input(
                    "How much", min_value=0.01, value=float(raw["size"]), step=1.0,
                    key=f"{prefix}_r{rid}_size", label_visibility="visible",
                )
            )
            raw["size_mode"] = st.selectbox(
                "Of", ar.SIZE_MODES, index=ar.SIZE_MODES.index(raw["size_mode"]),
                format_func=lambda m: ar.SIZE_LABEL[m], key=f"{prefix}_r{rid}_size_mode",
                help=(
                    "A percentage is of the cash balance on a buy and of the open "
                    "position on a sell. A dollar amount spends that much or sells that "
                    "much stock. Every mode is clipped to what the ledger can actually "
                    "do, so over-asking is safe."
                ),
            )
            raw["enabled"] = st.toggle(
                "On", value=bool(raw["enabled"]), key=f"{prefix}_r{rid}_enabled",
                help="A rule that is off is not checked, costs nothing, and does not "
                     "change how the run is filed in Results.",
            )

        with st.container(horizontal=True, vertical_alignment="bottom"):
            raw["join"] = (
                st.segmented_control(
                    "Fires when", ar.JOINS, default=raw["join"],
                    format_func=lambda j: _JOIN_LABEL[j], key=f"{prefix}_r{rid}_join",
                    help="One joiner per rule, on purpose: 'A and B or C' has no meaning "
                         "without precedence rules. A mix is two rules.",
                )
                or raw["join"]
            )
            raw["label"] = st.text_input(
                "Name (optional)", value=raw["label"], key=f"{prefix}_r{rid}_label",
                placeholder="e.g. trailing stop",
                help="Carried into the ledger, so a fill can be read back to the rule "
                     "that caused it.",
            )
            raw["cooldown_bars"] = int(
                st.number_input(
                    "Then wait (bars)", min_value=0, max_value=390,
                    value=int(raw["cooldown_bars"]), step=1,
                    key=f"{prefix}_r{rid}_cooldown",
                    help=(
                        "Bars this rule sits out after it fires. 0 lets it fire on every "
                        "bar its conditions hold, which ladders a balance in; 390 is one "
                        "session, so it fires once a day. Cooldowns reset each morning."
                    ),
                )
            )

        for cond_index, cond in enumerate(raw["conditions"]):
            _condition_editor(prefix, index, cond_index, cond, ticker)

        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.button(
                "Add condition", icon=":material/add:",
                key=f"{prefix}_r{rid}_add_cond", on_click=_add_condition,
                args=(prefix, index),
            )
            st.button(
                "Move up", icon=":material/arrow_upward:", key=f"{prefix}_r{rid}_up",
                disabled=index == 0, on_click=_move_rule, args=(prefix, index, -1),
                help="Order is precedence: the first rule that matches and can transact "
                     "takes the bar.",
            )
            st.button(
                "Move down", icon=":material/arrow_downward:", key=f"{prefix}_r{rid}_down",
                disabled=index == total - 1, on_click=_move_rule, args=(prefix, index, 1),
            )
            st.button(
                "Delete", icon=":material/delete:", key=f"{prefix}_r{rid}_del",
                on_click=_drop_rule, args=(prefix, index),
            )


def _condition_editor(
    prefix: str, index: int, cond_index: int, cond: dict, ticker: str
) -> None:
    """One condition. Flags get true/false instead of a number.

    The options are the signals this instrument has, plus -- if the condition
    already names one it does not -- that signal, kept in the list so switching
    the instrument shows what is wrong instead of silently rewriting the rule to
    something else. `_verdict` refuses the set while one is still there.
    """
    cid = cond["id"]
    keys = list(ar.signals_for(ticker))
    stale = cond["field"] in ar.SIGNALS and cond["field"] not in keys
    if stale:
        keys = [cond["field"], *keys]
    current = cond["field"] if cond["field"] in keys else keys[0]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        cond["field"] = st.selectbox(
            "Signal", keys, index=keys.index(current),
            format_func=lambda key: _signal_option(key, ticker),
            # Instrument-scoped for the same reason the preset box is: the
            # option list changes with it.
            key=f"{prefix}_c{cid}_field_{ticker}",
            label_visibility="collapsed" if cond_index else "visible",
        )
        spec = ar.SIGNALS[cond["field"]]
        if spec.flag:
            truth = st.segmented_control(
                "Is", (True, False), default=cond["op"] == ar.OP_ABOVE,
                format_func=lambda t: "is true" if t else "is false",
                key=f"{prefix}_c{cid}_flag",
                label_visibility="collapsed" if cond_index else "visible",
            )
            cond["op"] = ar.OP_ABOVE if truth is not False else ar.OP_BELOW
            cond["value"] = 0.5
        else:
            cond["op"] = st.selectbox(
                "Is", ar.OPS, index=ar.OPS.index(cond["op"]),
                format_func=lambda o: _OP_LABEL[o], key=f"{prefix}_c{cid}_op",
                label_visibility="collapsed" if cond_index else "visible",
            )
            cond["value"] = float(
                st.number_input(
                    f"Value ({spec.unit})" if spec.unit else "Value",
                    value=float(cond["value"]), step=_STEP.get(spec.unit, 1.0),
                    format=_FORMAT.get(spec.unit), key=f"{prefix}_c{cid}_value",
                    help=spec.help,
                    label_visibility="collapsed" if cond_index else "visible",
                )
            )
        st.button(
            "", icon=":material/close:", key=f"{prefix}_c{cid}_del",
            on_click=_drop_condition, args=(prefix, index, cond_index),
            help="Remove this condition.",
        )
    if not ar.readable_on(cond["field"], ticker):
        model = apple_models.get(ar.SIGNALS[cond["field"]].model)
        st.caption(
            f":material/error: `{cond['field']}` does not exist on {ticker} — "
            f"{model.label} was fitted on {', '.join(model.tickers)} only. Pick another "
            "signal or delete this condition."
        )
        return
    st.caption(f":material/info: {spec.label} — {spec.help}")


def _signal_option(key: str, ticker: str) -> str:
    spec = ar.SIGNALS[key]
    mark = "" if ar.readable_on(key, ticker) else f" — not available on {ticker}"
    return f"{spec.group} · {spec.label}  ({key}){mark}"


def _ruleset_from(rules: "list[dict]") -> ar.RuleSet:
    """The editor's dicts as a `RuleSet`, skipping anything half-built.

    A rule the user is mid-way through writing must not take the page down, so
    an item that cannot be constructed is dropped here and reported by
    `_verdict` rather than raised.
    """
    items = []
    for raw in rules:
        try:
            items.append(
                ar.ActionItem(
                    action=raw["action"],
                    size_mode=raw["size_mode"],
                    size=raw["size"],
                    join=raw["join"],
                    label=raw["label"],
                    enabled=raw["enabled"],
                    cooldown_bars=raw["cooldown_bars"],
                    conditions=[
                        ar.Condition(c["field"], c["op"], c["value"])
                        for c in raw["conditions"]
                    ],
                )
            )
        except (KeyError, ValueError):
            continue
    return ar.RuleSet(items=items)


def _verdict(config: AppleTrader2Config) -> None:
    """What these rules say, whether they can run, and what they will cost.

    The rendered list is the point of it: a builder that only shows widgets
    makes the reader reconstruct the strategy from twelve inputs, and the one
    thing worth checking before launching a rule set is that it reads the way it
    was meant to.
    """
    lines = ar.describe(config.rules)
    st.markdown(
        "**These rules, in the order they are checked** — the first that matches *and* "
        "can transact takes the bar:\n"
        + ("\n".join(f"1. `{line.split('. ', 1)[1]}`" for line in lines) if lines
           else "\n_Nothing yet._")
    )

    needed = config.rules.models()
    bundles = {key: apple_models.load(key, config.ticker) for key in needed}
    if needed:
        st.caption(
            f":material/memory: Models these rules load for {config.ticker}: "
            + ", ".join(apple_models.get(k).label for k in needed)
            + ". Every other bar signal is free."
        )
    else:
        st.caption(
            ":material/memory: These rules name no model, so nothing is loaded and the "
            "agent runs with no saved artifacts at all."
        )

    error = ar.ruleset_error(config.rules, bundles, config.ticker)
    if error:
        st.error(error)
    else:
        st.success(
            f"Runnable on {config.ticker}. Filed in Results as "
            f"`{ar.signature(config.rules, config.ticker)}`."
        )


def signal_catalogue(ticker: "str | None" = None) -> None:
    """The vocabulary, grouped -- rendered wherever the agent is described.

    With no ticker this is the whole catalogue (what the agent *can* read, on
    the instrument that has every model); with one it is what that instrument
    offers.
    """
    for group in ar.SIGNAL_GROUPS:
        rows = ar.signals_in(group, ticker)
        if not rows:
            continue
        with st.expander(f"{group} — {len(rows)} signals"):
            st.dataframe(
                [
                    {"Signal": s.key, "Reads": s.label, "Unit": s.unit, "Meaning": s.help}
                    for s in rows
                ],
                hide_index=True,
                width="stretch",
            )
