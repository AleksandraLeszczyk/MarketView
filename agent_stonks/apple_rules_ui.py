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


def _rules(prefix: str) -> "list[dict]":
    key = _state_key(prefix)
    if key not in st.session_state:
        st.session_state[key] = _with_ids(prefix, ar.preset())
    return st.session_state[key]


def load_preset(prefix: str, name: str) -> None:
    """Replace the editor's contents with a named preset."""
    st.session_state[_state_key(prefix)] = _with_ids(prefix, ar.preset(name))


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
    prefix: str, *, defaults: "AppleTrader2Config | None" = None
) -> AppleTrader2Config:
    """Render the builder and return the configuration it currently describes.

    Called on every rerun, so it both draws the widgets and reads them back: the
    editor's dicts are updated in place from the widget values, which is what
    keeps an edit alive across the rerun that an add or a delete causes.
    """
    if defaults is not None and _state_key(prefix) not in st.session_state:
        set_rules(prefix, defaults.rules)
    flatten_default = (defaults or AppleTrader2Config()).flatten_before_close_min

    _preset_row(prefix)
    rules = _rules(prefix)
    for index, raw in enumerate(rules):
        _rule_editor(prefix, index, raw, len(rules))

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
        rules=_ruleset_from(rules), flatten_before_close_min=int(flatten)
    )
    _verdict(config)
    return config


def _preset_row(prefix: str) -> None:
    names = list(ar.PRESETS)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        name = st.selectbox(
            "Start from", names, index=names.index(ar.DEFAULT_PRESET),
            key=f"{prefix}_preset",
            help=(
                "The first three are the strategies Apple Trader 1 ships, written out as "
                "rules — load one to see how a shipped strategy decomposes, or to have "
                "something measured to edit away from. Loading replaces everything below."
            ),
        )
        st.button(
            "Load", icon=":material/download:", key=f"{prefix}_load_preset",
            on_click=load_preset, args=(prefix, name),
        )


def _rule_editor(prefix: str, index: int, raw: dict, total: int) -> None:
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
            _condition_editor(prefix, index, cond_index, cond)

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


def _condition_editor(prefix: str, index: int, cond_index: int, cond: dict) -> None:
    """One condition. Flags get true/false instead of a number."""
    cid = cond["id"]
    keys = list(ar.SIGNALS)
    current = cond["field"] if cond["field"] in ar.SIGNALS else keys[0]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        cond["field"] = st.selectbox(
            "Signal", keys, index=keys.index(current),
            format_func=_signal_option, key=f"{prefix}_c{cid}_field",
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
    st.caption(f":material/info: {spec.label} — {spec.help}")


def _signal_option(key: str) -> str:
    spec = ar.SIGNALS[key]
    return f"{spec.group} · {spec.label}  ({key})"


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
    bundles = {key: apple_models.load(key) for key in needed}
    if needed:
        st.caption(
            ":material/memory: Models these rules load: "
            + ", ".join(apple_models.get(k).label for k in needed)
            + ". Every other bar signal is free."
        )
    else:
        st.caption(
            ":material/memory: These rules name no model, so nothing is loaded and the "
            "agent runs with no saved artifacts at all."
        )

    error = ar.ruleset_error(config.rules, bundles)
    if error:
        st.error(error)
    else:
        st.success(
            f"Runnable. Filed in Results as `{ar.signature(config.rules, 'AAPL')}`."
        )


def signal_catalogue() -> None:
    """The whole vocabulary, grouped -- rendered wherever the agent is described."""
    for group in ar.SIGNAL_GROUPS:
        rows = ar.signals_in(group)
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
