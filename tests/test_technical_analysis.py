import pandas as pd

from agent_stonks.technical_analysis import (
    adx,
    analyze_consolidation,
    analyze_fair_value_gaps,
    analyze_intraday,
    analyze_liquidity,
    analyze_market,
    analyze_order_blocks,
    analyze_premium_discount,
    analyze_smart_money_setup,
    analyze_trend,
    analyze_volume,
    analyze_volume_profile_2,
    analyze_vwap_bands,
    atr,
    breakout_trade_geometry,
    detect_regime_shift,
    find_fair_value_gaps,
    find_order_blocks,
    floor_pivots,
    get_put_call_walls_and_gamma,
    key_levels,
    obv_trend,
    piecewise_regimes,
    rsi,
    session_time_window,
    smart_money_trade_geometry,
    sma,
    support_resistance,
    swing_levels,
    volume_profile_levels,
    vwap_reversion_geometry,
)


def _make_bars(closes, highs=None, lows=None, volumes=None, vwaps=None):
    highs = highs or [c + 0.5 for c in closes]
    lows = lows or [c - 0.5 for c in closes]
    volumes = volumes or [1000] * len(closes)
    bars = []
    for i, c in enumerate(closes):
        bar = {"o": c, "h": highs[i], "l": lows[i], "c": c, "v": volumes[i]}
        if vwaps is not None:
            bar["vw"] = vwaps[i]
        bars.append(bar)
    return bars


class TestIndicatorHelpers:
    def test_sma_basic(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert sma(series, 3) == 4.0

    def test_sma_returns_none_when_not_enough_data(self):
        series = pd.Series([1.0, 2.0])
        assert sma(series, 3) is None

    def test_rsi_is_100_for_strictly_increasing_series(self):
        series = pd.Series([float(i) for i in range(10)])
        assert rsi(series, period=5) == 100.0

    def test_rsi_returns_none_when_not_enough_data(self):
        series = pd.Series([1.0, 2.0])
        assert rsi(series, period=5) is None

    def test_atr_equals_constant_high_low_range_when_close_unchanged(self):
        closes = [100.0] * 16
        bars = _make_bars(closes, highs=[101.0] * 16, lows=[99.0] * 16)
        assert atr(bars, period=14) == 2.0

    def test_atr_returns_none_when_not_enough_data(self):
        bars = _make_bars([100.0, 101.0])
        assert atr(bars, period=14) is None

    def test_obv_trend_rising_when_price_climbs_on_volume(self):
        closes = [100.0 + i for i in range(12)]
        bars = _make_bars(closes)
        assert obv_trend(bars, window=5) == "rising"

    def test_obv_trend_falling_when_price_drops_on_volume(self):
        closes = [100.0 - i for i in range(12)]
        bars = _make_bars(closes)
        assert obv_trend(bars, window=5) == "falling"

    def test_support_resistance_uses_lookback_window(self):
        closes = list(range(1, 31))
        bars = _make_bars([float(c) for c in closes])
        levels = support_resistance(bars, lookback=5)
        assert levels["lookback_bars"] == 5
        assert levels["support"] == bars[-5]["l"]
        assert levels["resistance"] == bars[-1]["h"]


class TestAnalyzeTrend:
    def test_not_enough_bars_returns_note(self):
        bars = _make_bars([100.0, 101.0])
        assert "note" in analyze_trend(bars)

    def test_steady_uptrend_is_classified_bullish(self):
        closes = [100.0 + i * 0.5 for i in range(70)]
        bars = _make_bars(closes)
        result = analyze_trend(bars)
        assert result["regime"] == "bullish"
        assert result["pct_change_over_period"] > 0
        assert "summary" in result

    def test_steady_downtrend_is_classified_bearish(self):
        closes = [150.0 - i * 0.5 for i in range(70)]
        bars = _make_bars(closes)
        result = analyze_trend(bars)
        assert result["regime"] == "bearish"
        assert result["pct_change_over_period"] < 0

    def test_flat_choppy_series_is_classified_neutral(self):
        closes = [100.0, 100.5, 99.8, 100.2, 99.9, 100.1, 100.0, 99.7, 100.3, 100.0] * 6
        bars = _make_bars(closes)
        result = analyze_trend(bars)
        assert result["regime"] == "neutral"


class TestAnalyzeIntraday:
    def test_not_enough_bars_returns_note(self):
        bars = _make_bars([100.0, 101.0])
        assert "note" in analyze_intraday(bars)

    def test_detects_higher_highs_and_higher_lows(self):
        closes = [100.0 + i for i in range(20)]
        bars = _make_bars(closes)
        result = analyze_intraday(bars)
        assert "uptrend" in result["momentum_pattern"]

    def test_detects_lower_highs_and_lower_lows(self):
        closes = [120.0 - i for i in range(20)]
        bars = _make_bars(closes)
        result = analyze_intraday(bars)
        assert "downtrend" in result["momentum_pattern"]

    def test_vwap_position_reported_when_present(self):
        closes = [100.0] * 10
        bars = _make_bars(closes, vwaps=[99.0] * 10)
        result = analyze_intraday(bars)
        assert "above" in result["vwap_position"]

    def test_yesterday_momentum_block_from_prev_session(self):
        today = _make_bars([100.0 + i for i in range(20)])
        yesterday = _make_bars([120.0 - i for i in range(20)])  # down day
        result = analyze_intraday(today, prev_session_bars=yesterday)
        block = result["yesterday_momentum"]
        assert block is not None
        assert block["pct_change"] < 0
        assert "downtrend" in block["momentum_pattern"]

    def test_today_total_momentum_uses_full_session_not_window(self):
        # Recent window is a fade off the highs, but the full session is a big up day.
        full = _make_bars([100.0 + i for i in range(30)] + [130.0 - i * 0.2 for i in range(10)])
        window = full[-10:]
        result = analyze_intraday(window, full_session_bars=full)
        assert result["today_total_momentum"]["pct_change"] > 0
        assert result["today_total_momentum"]["bars"] == len(full)

    def test_current_momentum_duration_reports_last_leg(self):
        # A clean down leg followed by a clean up leg: current leg should be up.
        closes = [120.0 - i for i in range(20)] + [100.0 + i for i in range(20)]
        bars = _make_bars(closes)
        result = analyze_intraday(bars, full_session_bars=bars)
        leg = result["current_momentum_duration"]
        assert leg is not None
        assert leg["direction"] == "up"
        assert leg["bars"] >= 3
        assert leg["regimes_in_session"] >= 2

    def test_market_neutral_strips_beta_scaled_market_move(self):
        # Ticker up 2% over the window; market up 2% with beta 1.0 -> ~0 residual.
        closes = [100.0 * (1 + 0.02 * i / 19) for i in range(20)]
        bars = _make_bars(closes)
        result = analyze_intraday(bars, market_return_pct=2.0, beta=1.0)
        mn = result["market_neutral_momentum"]
        assert mn is not None
        assert mn["beta"] == 1.0
        assert abs(mn["residual_pct"]) < 0.2
        assert mn["market_component_pct"] == 2.0

    def test_market_neutral_absent_without_inputs(self):
        bars = _make_bars([100.0 + i for i in range(20)])
        result = analyze_intraday(bars)
        assert result["market_neutral_momentum"] is None


class TestPiecewiseRegimes:
    def test_single_straight_line_is_one_regime(self):
        closes = pd.Series([100.0 + i for i in range(30)])
        regimes = piecewise_regimes(closes)
        assert len(regimes) == 1
        assert regimes[0]["direction"] == "up"
        assert regimes[0]["bars"] == 30

    def test_v_shape_splits_into_down_then_up(self):
        closes = pd.Series([120.0 - i for i in range(20)] + [100.0 + i for i in range(20)])
        regimes = piecewise_regimes(closes)
        assert len(regimes) >= 2
        assert regimes[0]["direction"] == "down"
        assert regimes[-1]["direction"] == "up"

    def test_flat_series_is_flat_direction(self):
        closes = pd.Series([100.0] * 30)
        regimes = piecewise_regimes(closes)
        assert all(r["direction"] == "flat" for r in regimes)


class TestAnalyzeMarket:
    def test_no_data_returns_note(self):
        assert "note" in analyze_market(vix_close=None, spy_close=None)

    def test_calm_vix_and_uptrending_spy_is_risk_on(self):
        vix = pd.Series([13.0] * 10)
        spy = pd.Series([300.0 + i for i in range(260)])  # steady uptrend, above 50/200d
        result = analyze_market(vix_close=vix, spy_close=spy)
        assert result["risk_environment"] == "risk-on"
        assert result["risk_score"] > 0
        assert result["vix_label"] == "low (calm)"
        assert result["insights"]

    def test_high_vix_and_falling_spy_is_risk_off(self):
        vix = pd.Series([38.0] * 10)
        spy = pd.Series([500.0 - i for i in range(260)])  # steady downtrend, below 50/200d
        result = analyze_market(vix_close=vix, spy_close=spy)
        assert result["risk_environment"] == "risk-off"
        assert result["risk_score"] < 0
        assert "extreme (panic)" in result["vix_label"]

    def test_inverted_term_structure_flagged_as_backwardation(self):
        vix = pd.Series([30.0] * 10)
        vix3m = pd.Series([26.0] * 10)
        result = analyze_market(vix_close=vix, vix3m_close=vix3m, spy_close=None)
        assert result["vix_term_structure"] == "backwardation"
        assert any("inverted" in i for i in result["insights"])

    def test_contango_term_structure_when_vix3m_above_vix(self):
        vix = pd.Series([18.0] * 10)
        vix3m = pd.Series([20.0] * 10)
        result = analyze_market(vix_close=vix, vix3m_close=vix3m, spy_close=None)
        assert result["vix_term_structure"] == "contango"

    def test_spy_drawdown_reported(self):
        # Peak at 100, ends 15% lower -> correction territory.
        spy = pd.Series([100.0] * 5 + [85.0])
        result = analyze_market(vix_close=None, spy_close=spy)
        assert result["spy_drawdown_from_high_pct"] == -15.0
        assert any("correction" in i for i in result["insights"])

    def test_rising_vix_adds_caution_insight(self):
        vix = pd.Series([20.0, 21.0, 22.0, 24.0, 26.0, 28.0])  # +40% over 5 sessions
        result = analyze_market(vix_close=vix, spy_close=None)
        assert result["vix_5d_change_pct"] > 10
        assert any("rising fast" in i for i in result["insights"])


class TestAnalyzeVolume:
    def test_no_bars_returns_note(self):
        assert "note" in analyze_volume([])

    def test_rising_price_and_volume_is_confirming(self):
        closes = [100.0 + i for i in range(20)]
        volumes = [1000 + i * 100 for i in range(20)]
        bars = _make_bars(closes, volumes=volumes)
        result = analyze_volume(bars)
        assert result["volume_trend"] == "increasing"
        assert result["confirmation"] == "confirming"

    def test_rising_price_with_falling_volume_is_diverging(self):
        closes = [100.0 + i for i in range(20)]
        volumes = [5000 - i * 200 for i in range(20)]
        bars = _make_bars(closes, volumes=volumes)
        result = analyze_volume(bars)
        assert result["volume_trend"] == "decreasing"
        assert "diverging" in result["confirmation"]

    def test_participation_gate_needs_both_pace_and_local_volume(self):
        closes = [100.0 + i for i in range(20)]
        # Flat volume: local relative_volume lands at ~1.0, under the 1.2 floor.
        flat = _make_bars(closes, volumes=[1000] * 20)
        # High session pace on its own must NOT pass -- this is the exact
        # combination that waved through every losing tactic entry.
        assert analyze_volume(flat, rvol_pace=3.0)["participation_ok"] is False

        # Local participation doubling, but a quiet session overall.
        rising = _make_bars(closes, volumes=[1000] * 10 + [2000] * 10)
        assert analyze_volume(rising, rvol_pace=1.1)["participation_ok"] is False

        # Both clear.
        result = analyze_volume(rising, rvol_pace=2.5)
        assert result["participation_ok"] is True
        assert "PASS" in result["summary"]

    def test_participation_gate_is_none_without_pace(self):
        bars = _make_bars([100.0 + i for i in range(20)], volumes=[1000] * 20)
        assert analyze_volume(bars)["participation_ok"] is None

    def test_volume_burst_measures_the_newest_bars(self):
        closes = [100.0 + i for i in range(20)]
        # Last 3 bars run hot against the 10-bar average behind them.
        volumes = [1000] * 17 + [4000] * 3
        result = analyze_volume(_make_bars(closes, volumes=volumes))
        assert result["volume_burst"] > 1.5
        quiet = analyze_volume(_make_bars(closes, volumes=[1000] * 20))
        assert quiet["volume_burst"] == 1.0


class TestGetPutCallWallsAndGamma:
    def test_no_strikes_returns_note(self):
        assert "note" in get_put_call_walls_and_gamma([], [], [], [], [], spot=100.0)

    def test_identifies_call_wall_and_put_wall_from_peak_open_interest(self):
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        calls_oi = [100, 200, 300, 400, 1000]
        puts_oi = [900, 300, 200, 100, 50]
        gamma = [0.0] * 5
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, gamma, gamma, spot=100.0)
        assert result["call_wall"] == 110.0
        assert result["put_wall"] == 90.0
        assert result["in_range"] is True

    def test_positive_net_gamma_is_dampening_regime(self):
        strikes = [95.0, 100.0, 105.0]
        calls_oi = [100, 100, 100]
        puts_oi = [100, 100, 100]
        calls_gamma = [10.0, 10.0, 10.0]
        puts_gamma = [-1.0, -1.0, -1.0]
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, calls_gamma, puts_gamma, spot=100.0)
        assert result["net_gamma"] > 0
        assert "positive" in result["gamma_regime"]

    def test_negative_net_gamma_is_amplifying_regime(self):
        strikes = [95.0, 100.0, 105.0]
        calls_oi = [100, 100, 100]
        puts_oi = [100, 100, 100]
        calls_gamma = [1.0, 1.0, 1.0]
        puts_gamma = [-10.0, -10.0, -10.0]
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, calls_gamma, puts_gamma, spot=100.0)
        assert result["net_gamma"] < 0
        assert "negative" in result["gamma_regime"]

    def test_gamma_flip_found_between_negative_and_positive_strikes(self):
        strikes = [95.0, 100.0, 105.0]
        calls_oi = puts_oi = [100, 100, 100]
        calls_gamma = [0.0, 5.0, 20.0]
        puts_gamma = [-10.0, 0.0, 0.0]
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, calls_gamma, puts_gamma, spot=100.0)
        assert result["gamma_flip"] is not None
        assert 100.0 < result["gamma_flip"] < 105.0

    def test_spot_above_call_wall_flags_breach(self):
        strikes = [90.0, 95.0, 100.0]
        calls_oi = [500, 100, 50]
        puts_oi = [50, 100, 500]
        gamma = [0.0] * 3
        result = get_put_call_walls_and_gamma(strikes, calls_oi, puts_oi, gamma, gamma, spot=110.0)
        assert result["in_range"] is False
        assert any("above the call wall" in i for i in result["insights"])

    def test_rising_call_wall_trend_detected_from_history(self):
        strikes = [95.0, 100.0, 105.0]
        calls_oi = [100, 100, 500]
        puts_oi = [500, 100, 100]
        gamma = [0.0] * 3
        history = [{"call_wall": 95.0, "put_wall": 90.0}, {"call_wall": 100.0, "put_wall": 90.0}]
        result = get_put_call_walls_and_gamma(
            strikes, calls_oi, puts_oi, gamma, gamma, spot=100.0, wall_history=history
        )
        assert result["call_wall_trend"] == "rising"
        assert any("rising" in i for i in result["insights"])


class TestAnalyzeConsolidation:
    def test_not_enough_bars_returns_note(self):
        bars = _make_bars([100.0, 101.0])
        assert "note" in analyze_consolidation(bars)

    def test_contracting_range_and_declining_volume_reads_as_coiling(self):
        prior_closes = [90.0, 110.0] * 10  # wide swings -> wide prior range
        prior_bars = _make_bars(prior_closes, highs=[c + 10 for c in prior_closes], lows=[c - 10 for c in prior_closes], volumes=[2000] * 20)
        base_closes = [100.0] * 10  # tight range
        base_bars = _make_bars(base_closes, highs=[100.5] * 10, lows=[99.5] * 10, volumes=[500] * 10)
        result = analyze_consolidation(prior_bars + base_bars, base_bars=10, prior_bars=20)

        assert result["base_high"] == 100.5
        assert result["base_low"] == 99.5
        assert result["base_height"] == 1.0
        assert result["range_contraction_pct"] > 10
        assert result["volume_trend_in_base"] == "declining"
        assert result["is_coiling"] is True

    def test_expanding_range_is_not_coiling(self):
        prior_closes = [100.0] * 20
        prior_bars = _make_bars(prior_closes, highs=[100.5] * 20, lows=[99.5] * 20, volumes=[500] * 20)
        base_closes = [90.0, 110.0] * 5
        base_bars = _make_bars(base_closes, highs=[c + 10 for c in base_closes], lows=[c - 10 for c in base_closes], volumes=[2000] * 10)
        result = analyze_consolidation(prior_bars + base_bars, base_bars=10, prior_bars=20)

        assert result["is_coiling"] is False

    def test_counts_touches_at_resistance_and_support(self):
        prior_bars = _make_bars([100.0] * 20, volumes=[1000] * 20)
        highs = [100.5, 99.0, 100.5, 99.0, 100.5, 99.0, 99.0, 99.0, 99.0, 99.0]
        lows = [99.5, 98.5, 99.5, 98.5, 99.5, 98.5, 98.5, 98.5, 98.5, 98.5]
        base_bars = _make_bars([99.0] * 10, highs=highs, lows=lows, volumes=[1000] * 10)
        result = analyze_consolidation(prior_bars + base_bars, base_bars=10, prior_bars=20)

        assert result["touches_at_resistance"] == 3
        assert result["well_tested"] is True


class TestSessionTimeWindow:
    def test_opening_window_is_favorable(self):
        result = session_time_window("2024-07-15T13:35:00Z")  # 09:35 ET (EDT, UTC-4)
        assert result["window"] == "opening_window"
        assert result["favorable_for_breakouts"] is True

    def test_midday_dead_zone_is_unfavorable(self):
        result = session_time_window("2024-07-15T17:30:00Z")  # 13:30 ET
        assert result["window"] == "midday_dead_zone"
        assert result["favorable_for_breakouts"] is False

    def test_power_hour_is_favorable(self):
        result = session_time_window("2024-07-15T19:45:00Z")  # 15:45 ET
        assert result["window"] == "power_hour"
        assert result["favorable_for_breakouts"] is True

    def test_outside_regular_hours_is_unfavorable(self):
        result = session_time_window("2024-07-15T03:00:00Z")  # 23:00 ET prior day
        assert result["window"] == "outside_regular_hours"
        assert result["favorable_for_breakouts"] is False

    def test_falls_back_to_now_when_no_timestamp_given(self):
        result = session_time_window(None)
        assert "et_time" in result
        assert "window" in result


def _osc_bars(n=40, center=100.0, amp=1.0, vol=1000):
    """Choppy, alternating bars -- a ranging tape (balanced +DI/-DI, low ADX)."""
    bars = []
    for i in range(n):
        c = center + (amp if i % 2 == 0 else -amp)
        bars.append({"o": c, "h": c + 0.3, "l": c - 0.3, "c": c, "v": vol})
    return bars


class TestADX:
    def test_returns_none_when_not_enough_bars(self):
        assert adx(_make_bars([100.0] * 10), period=14) is None

    def test_strong_uptrend_reads_as_trending(self):
        closes = [100.0 + i for i in range(40)]
        assert adx(_make_bars(closes), period=14) >= 25

    def test_choppy_range_reads_as_non_trending(self):
        # Balanced up/down movement -> +DI and -DI cancel -> low ADX.
        assert adx(_osc_bars(40), period=14) < 25


class TestAnalyzeVwapBands:
    def test_not_enough_bars_returns_note(self):
        assert "note" in analyze_vwap_bands(_make_bars([100.0, 101.0]))

    def test_bands_are_ordered_around_vwap(self):
        result = analyze_vwap_bands(_osc_bars(40))
        assert result["lower_band_3sd"] < result["lower_band_2sd"] < result["lower_band_1sd"]
        assert result["lower_band_1sd"] < result["vwap"] < result["upper_band_1sd"]
        assert result["upper_band_1sd"] < result["upper_band_2sd"] < result["upper_band_3sd"]

    def test_oversold_stretch_in_range_is_long_setup(self):
        bars = _osc_bars(40)
        # A deep dip on the final bar with a long lower wick (bullish rejection).
        bars.append({"o": 97.0, "h": 97.2, "l": 95.5, "c": 97.0, "v": 1000})
        result = analyze_vwap_bands(bars, num_std=2.0)
        assert result["z_score"] <= -2.0
        assert result["is_ranging"] is True
        assert result["signal"] == "long_setup"
        assert result["rejection_candle"] == "bullish_rejection"

    def test_strong_trend_is_not_a_fade_setup(self):
        closes = [100.0 + i for i in range(40)]
        result = analyze_vwap_bands(_make_bars(closes), num_std=2.0)
        assert result["is_ranging"] is False
        assert result["signal"] in ("no_setup", "no_setup_trending")


class TestVwapReversionGeometry:
    def test_non_positive_inputs_return_note(self):
        assert "note" in vwap_reversion_geometry(entry=0, vwap=100.0, std_dev=1.0)

    def test_long_entry_must_be_below_vwap(self):
        assert "note" in vwap_reversion_geometry(entry=101.0, vwap=100.0, std_dev=1.0, side="long")

    def test_long_reversion_geometry_and_reward_risk(self):
        # Entry 2sd below VWAP, stop 1sd further down, target VWAP -> 2:1.
        result = vwap_reversion_geometry(entry=98.0, vwap=100.0, std_dev=1.0, side="long")
        assert result["stop"] == 97.0
        assert result["target"] == 100.0
        assert result["risk_per_share"] == 1.0
        assert result["reward_per_share"] == 2.0
        assert result["reward_risk_ratio"] == 2.0
        assert result["meets_min_reward_risk"] is True

    def test_shallow_stretch_fails_min_reward_risk(self):
        # Entry only 1sd below VWAP -> reward == risk -> 1:1, below the 1.5 minimum.
        result = vwap_reversion_geometry(entry=99.0, vwap=100.0, std_dev=1.0, side="long")
        assert result["reward_risk_ratio"] == 1.0
        assert result["meets_min_reward_risk"] is False

    def test_short_reversion_geometry(self):
        result = vwap_reversion_geometry(entry=102.0, vwap=100.0, std_dev=1.0, side="short")
        assert result["stop"] == 103.0
        assert result["risk_per_share"] == 1.0
        assert result["reward_per_share"] == 2.0
        assert result["meets_min_reward_risk"] is True


def _ts_bar(t, o, h, l, c, v=1000):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}


class TestKeyLevels:
    # July is EDT (UTC-4): the 9:30 ET bell is 13:30Z.
    PRIOR_DAILY = _ts_bar("2026-07-16T04:00:00Z", 100.0, 110.0, 95.0, 105.0)
    TODAY_DAILY = _ts_bar("2026-07-17T04:00:00Z", 102.0, 107.0, 101.0, 106.0)
    INTRADAY = [
        _ts_bar("2026-07-17T13:00:00Z", 102.0, 103.0, 101.0, 102.5),  # premarket (9:00 ET)
        _ts_bar("2026-07-17T13:30:00Z", 102.0, 104.0, 101.5, 103.5),
        _ts_bar("2026-07-17T13:31:00Z", 103.5, 106.0, 103.0, 105.5),
        _ts_bar("2026-07-17T13:32:00Z", 105.5, 107.0, 105.0, 106.0),
    ]

    def test_no_data_returns_note(self):
        assert "note" in key_levels([], daily_bars=None)

    def test_collects_session_structure_levels(self):
        result = key_levels(
            self.INTRADAY,
            daily_bars=[self.PRIOR_DAILY, self.TODAY_DAILY],
            opening_range_minutes=2,
        )
        levels = result["levels"]
        # Prior-day levels come from the last COMPLETED daily bar, not today's.
        assert levels["prior_day_high"] == 110.0
        assert levels["prior_day_low"] == 95.0
        assert levels["prior_day_close"] == 105.0
        assert levels["premarket_high"] == 103.0
        assert levels["opening_range_high"] == 106.0
        assert levels["opening_range_low"] == 101.5
        assert levels["session_high"] == 107.0
        assert levels["session_low"] == 101.5

    def test_splits_levels_around_spot_nearest_first(self):
        result = key_levels(
            self.INTRADAY,
            daily_bars=[self.PRIOR_DAILY, self.TODAY_DAILY],
            opening_range_minutes=2,
        )
        assert result["spot"] == 106.0
        assert result["nearest_resistance"]["name"] == "session_high"
        assert result["nearest_resistance"]["level"] == 107.0
        assert [e["name"] for e in result["resistance_above"]] == ["session_high", "prior_day_high"]
        support_names = [e["name"] for e in result["support_below"]]
        assert support_names[0] == "opening_range_high"
        assert "premarket_high" in support_names

    def test_spot_above_everything_is_blue_sky(self):
        result = key_levels(self.INTRADAY, daily_bars=[self.PRIOR_DAILY], spot=120.0)
        assert result["nearest_resistance"] is None
        assert result["resistance_above"] == []
        assert "blue-sky" in result["summary"]

    def test_timestampless_bars_count_as_session(self):
        bars = _make_bars([100.0, 101.0, 102.0])
        result = key_levels(bars)
        assert result["levels"]["session_high"] == 102.5
        assert result["levels"]["session_low"] == 99.5


class TestSwingLevels:
    def test_not_enough_bars_returns_note(self):
        assert "note" in swing_levels(_make_bars([100.0, 101.0]))

    def test_clusters_repeated_swing_highs_by_touch_count(self):
        highs = [101, 102, 103, 105, 103, 102, 101, 102, 103, 104, 105, 103, 102, 101, 100]
        lows = [99, 100, 101, 103, 101, 100, 99, 100, 101, 102, 103, 101, 100, 99, 98]
        closes = [100, 101, 102, 104, 102, 101, 100, 101, 102, 103, 104, 102, 101, 100, 99]
        result = swing_levels(_make_bars(closes, highs=[float(x) for x in highs], lows=[float(x) for x in lows]))
        # The double top at 105 (indices 3 and 10) merges into one 2-touch cluster,
        # ranked ahead of any single-touch level.
        assert result["levels"][0]["touches"] == 2
        assert result["levels"][0]["level"] == 105.0
        assert result["nearest_resistance"]["level"] == 105.0
        assert result["nearest_support"] is not None
        assert result["nearest_support"]["level"] <= result["spot"]

    def test_spot_above_all_swings_has_no_resistance(self):
        highs = [101.0, 102.0, 103.0, 105.0, 103.0, 102.0, 101.0, 102.0, 103.0]
        closes = [100.0, 101.0, 102.0, 104.0, 102.0, 101.0, 100.0, 101.0, 102.0]
        result = swing_levels(_make_bars(closes, highs=highs), spot=120.0)
        assert result["nearest_resistance"] is None


class TestVolumeProfileLevels:
    def test_not_enough_bars_returns_note(self):
        assert "note" in volume_profile_levels(_make_bars([100.0] * 5))

    def test_poc_sits_where_the_volume_traded(self):
        heavy = _make_bars([100.0] * 15, volumes=[10_000] * 15)
        ramp = _make_bars([102.0, 104.0, 106.0, 108.0, 110.0], volumes=[100] * 5)
        result = volume_profile_levels(heavy + ramp, bins=10)
        assert 99.0 <= result["poc"] <= 101.0
        # Spot (110, the last close) sits in the thin ramp: an air pocket, with
        # the heavy 100-area node below it.
        assert result["spot_in_low_volume_node"] is True
        assert result["nearest_hvn_above"] is None
        assert 99.0 <= result["nearest_hvn_below"]["price"] <= 101.0
        assert result["low_volume_nodes"]

    def test_value_area_covers_the_heavy_node(self):
        heavy = _make_bars([100.0] * 15, volumes=[10_000] * 15)
        ramp = _make_bars([102.0, 104.0, 106.0, 108.0, 110.0], volumes=[100] * 5)
        result = volume_profile_levels(heavy + ramp, bins=10)
        assert result["value_area_low"] <= 100.0 <= result["value_area_high"]


def _session_bars(closes, volumes, start="2026-07-17T13:30:00Z"):
    """Minute bars from `start` (default 09:30 ET on 2026-07-17), one per
    (close, volume) pair, with a tight high/low around each close."""
    from datetime import datetime, timedelta

    base = datetime.fromisoformat(start.replace("Z", "+00:00"))
    bars = []
    for i, (c, v) in enumerate(zip(closes, volumes)):
        t = base + timedelta(minutes=i)
        bars.append(
            {
                "t": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "o": float(c),
                "h": float(c) + 0.05,
                "l": float(c) - 0.05,
                "c": float(c),
                "v": float(v),
            }
        )
    return bars


class TestAnalyzeVolumeProfile2:
    def test_too_few_bars_returns_note(self):
        bars = _session_bars([100.0] * 20, [1000] * 20)
        assert "note" in analyze_volume_profile_2(bars)

    def test_supply_spike_is_resistance(self):
        # 15 warm-up min, then a run up into a heavy minute, then a fade down.
        closes = [100.0] * 15
        closes += [100.0 + 0.1 * i for i in range(1, 16)]   # up into the spike (idx 30)
        closes += [closes[-1] - 0.1 * i for i in range(1, 16)]  # down after
        vols = [5000] * 15 + [1000] * 30
        vols[30] = 6000  # the spike minute
        result = analyze_volume_profile_2(_session_bars(closes, vols))
        supply = [p for p in result["peaks"] if p["type"] == "supply"]
        assert supply, result["summary"]
        assert supply[0]["rel_vol"] >= 3.0

    def test_demand_spike_is_support(self):
        # Down into the heavy minute, then a rally out of it.
        closes = [100.0] * 15
        closes += [100.0 - 0.1 * i for i in range(1, 16)]   # down into the spike (idx 30)
        closes += [closes[-1] + 0.1 * i for i in range(1, 16)]  # up after
        vols = [5000] * 15 + [1000] * 30
        vols[30] = 6000
        result = analyze_volume_profile_2(_session_bars(closes, vols))
        demand = [p for p in result["peaks"] if p["type"] == "demand"]
        assert demand, result["summary"]

    def test_opening_warmup_spike_is_ignored(self):
        # The single heaviest minute is in the opening warm-up window; it must
        # not be reported as a spike.
        closes = [100.0 + 0.05 * i for i in range(60)]
        vols = [1000] * 60
        vols[5] = 50_000  # enormous, but inside the first 15 min
        result = analyze_volume_profile_2(_session_bars(closes, vols))
        assert all(p["time"] > "09:45" for p in result["peaks"] if p["type"] != "scattered")

    def test_news_driven_flag_and_pre_news_removal(self):
        # Two spikes: an early one, and a later one straddled by a news print.
        closes = [100.0] * 15 + [100.0 + 0.1 * i for i in range(1, 46)]
        vols = [5000] * 15 + [1000] * 45
        vols[25] = 6000  # early spike (~09:55 ET)
        vols[45] = 6000  # later spike (~10:15 ET)
        news = ["2026-07-17T14:15:00Z"]  # 10:15 ET, on the later spike
        result = analyze_volume_profile_2(_session_bars(closes, vols), news_times=news)
        assert result["removed_pre_news"] >= 1
        # Everything surviving is at or after the catalyst minute.
        assert all(p["time"] >= "10:15" for p in result["peaks"])
        assert any(p["news_driven"] for p in result["peaks"])

    def test_no_news_keeps_all_peaks(self):
        closes = [100.0] * 15 + [100.0 + 0.1 * i for i in range(1, 46)]
        vols = [5000] * 15 + [1000] * 45
        vols[25] = 6000
        vols[45] = 6000
        result = analyze_volume_profile_2(_session_bars(closes, vols), news_times=[])
        assert result["removed_pre_news"] == 0

    def test_peak_shape_has_required_fields(self):
        closes = [100.0] * 15 + [100.0 + 0.1 * i for i in range(1, 16)] + [
            103.0 - 0.1 * i for i in range(1, 16)
        ]
        vols = [5000] * 15 + [1000] * 30
        vols[30] = 6000
        result = analyze_volume_profile_2(_session_bars(closes, vols))
        assert result["peaks"]
        for p in result["peaks"]:
            assert set(p) >= {"price", "time", "date", "rel_vol", "vol", "type"}
            assert p["type"] in {"supply", "demand", "unsure", "scattered"}
            assert p["date"] == "2026-07-17"


class TestDetectRegimeShift:
    def test_too_few_bars_returns_note(self):
        assert "note" in detect_regime_shift(_session_bars([100.0] * 5, [1000] * 5))

    def test_steady_uptrend_has_no_shift_and_keeps_the_trend_bias(self):
        bars = _session_bars([100.0 + 0.08 * i for i in range(45)], [1000] * 45)
        result = detect_regime_shift(bars)
        assert result["regime"] == "trending_up"
        assert result["shift_detected"] is False
        assert result["bias"] == "trend_intact"
        assert result["levels_stale"] is False

    def test_rally_then_reversal_is_detected_as_bearish_with_giveback(self):
        # 30 minutes up, then a faster 15-minute slide on heavy volume: the
        # exact tape a level-only read keeps buying demand lines into.
        up = [100.0 + 0.10 * i for i in range(30)]
        down = [up[-1] - 0.14 * i for i in range(1, 16)]
        result = detect_regime_shift(_session_bars(up + down, [1000] * 30 + [3000] * 15))
        assert result["shift_detected"] is True
        assert result["shift_direction"] == "bearish"
        assert result["regime"] == "turning_down"
        assert result["bias"] == "exit_or_stand_aside"
        assert result["turn"]["transition"] == "reversal_down"
        assert result["giveback_pct_of_prior_leg"] > 50
        assert result["last_vwap_cross"]["direction"] == "lost_vwap"
        # The turn was carried by volume, so the pre-turn level map is stale.
        assert result["turn_volume_rel"] >= 1.5
        assert result["levels_stale"] is True

    def test_selloff_then_recovery_is_detected_as_bullish(self):
        down = [110.0 - 0.12 * i for i in range(30)]
        up = [down[-1] + 0.14 * i for i in range(1, 16)]
        result = detect_regime_shift(_session_bars(down + up, [1000] * 45))
        assert result["shift_direction"] == "bullish"
        assert result["regime"] == "turning_up"
        assert result["bias"] == "re_engage"
        assert result["turn"]["transition"] == "reversal_up"

    def test_thin_turn_without_volume_leaves_the_level_map_valid(self):
        # Same shape, but the turn drifts on below-median volume: a provisional
        # turn should not by itself void the levels.
        up = [100.0 + 0.10 * i for i in range(30)]
        down = [up[-1] - 0.14 * i for i in range(1, 16)]
        result = detect_regime_shift(_session_bars(up + down, [3000] * 30 + [800] * 15))
        assert result["shift_detected"] is True
        assert result["turn_volume_rel"] < 1.5
        assert result["levels_stale"] is False

    def test_news_at_the_turn_marks_the_level_map_stale(self):
        up = [100.0 + 0.10 * i for i in range(30)]
        down = [up[-1] - 0.14 * i for i in range(1, 16)]
        bars = _session_bars(up + down, [1000] * 45)
        # 10:00 ET, the minute the up leg ends and the slide begins.
        result = detect_regime_shift(bars, news_times=["2026-07-17T14:00:00Z"])
        assert result["news_at_turn"] is True
        assert result["levels_stale"] is True

    def test_decelerating_velocity_is_reported_before_the_leg_flips(self):
        # An advance that flattens out: velocity fades while the leg is still up.
        closes = [100.0 + 0.10 * i for i in range(30)] + [103.0 + 0.005 * i for i in range(1, 11)]
        result = detect_regime_shift(_session_bars(closes, [1000] * 40))
        assert result["velocity"]["state"] == "decelerating"
        assert result["velocity"]["recent_pct"] < result["velocity"]["previous_pct"]

    def test_reference_levels_are_present_for_arming_tactics(self):
        closes = [100.0 + 0.10 * i for i in range(30)] + [103.0 - 0.10 * i for i in range(1, 16)]
        result = detect_regime_shift(_session_bars(closes, [1000] * 45))
        levels = result["reference_levels"]
        assert set(levels) == {
            "price",
            "session_vwap",
            "turn_price",
            "last_swing_high",
            "last_swing_low",
            "session_high",
            "session_low",
        }
        assert levels["session_high"] >= levels["price"] >= levels["session_low"]
        assert levels["session_vwap"] is not None

    def test_bearish_structure_break_when_the_last_swing_low_gives_way(self):
        # Up, a pullback that prints a confirmed swing low, a bounce, then a
        # break of that low.
        closes = (
            [100.0 + 0.10 * i for i in range(20)]        # advance
            + [102.0 - 0.10 * i for i in range(1, 8)]    # pullback to ~101.3
            + [101.3 + 0.10 * i for i in range(1, 8)]    # bounce
            + [102.0 - 0.15 * i for i in range(1, 12)]   # break the swing low
        )
        result = detect_regime_shift(_session_bars(closes, [1000] * len(closes)))
        assert result["structure_break"] == "bearish"
        assert result["last_swing_low"]["price"] > result["reference_levels"]["price"]
        assert any("break of structure" in i for i in result["insights"])


class TestFloorPivots:
    PRIOR_DAILY = {"t": "2026-07-16T04:00:00Z", "o": 102.0, "h": 110.0, "l": 100.0, "c": 105.0, "v": 1e6}
    TODAY_DAILY = {"t": "2026-07-17T04:00:00Z", "o": 104.0, "h": 108.0, "l": 103.0, "c": 106.0, "v": 5e5}

    def test_no_prior_day_returns_note(self):
        assert "note" in floor_pivots([], today="2026-07-17")
        assert "note" in floor_pivots([self.TODAY_DAILY], today="2026-07-17")

    def test_computes_classic_pivot_formulas_from_prior_day(self):
        result = floor_pivots([self.PRIOR_DAILY, self.TODAY_DAILY], today="2026-07-17")
        levels = result["levels"]
        assert result["prior_day_date"] == "2026-07-16"
        assert levels["pivot"] == 105.0
        assert levels["r1"] == 110.0
        assert levels["s1"] == 100.0
        assert levels["r2"] == 115.0
        assert levels["s2"] == 95.0
        assert levels["r3"] == 120.0
        assert levels["s3"] == 90.0

    def test_splits_levels_around_spot(self):
        result = floor_pivots([self.PRIOR_DAILY], spot=106.0, today="2026-07-17")
        assert result["nearest_resistance"]["name"] == "r1"
        assert result["nearest_resistance"]["level"] == 110.0
        assert result["nearest_support"]["name"] == "pivot"


class TestBreakoutTradeGeometry:
    def test_non_positive_entry_or_stop_returns_note(self):
        assert "note" in breakout_trade_geometry(entry=0, stop=10)

    def test_stop_above_entry_returns_note(self):
        assert "note" in breakout_trade_geometry(entry=100.0, stop=105.0)

    def test_base_height_target_meets_min_reward_risk(self):
        result = breakout_trade_geometry(entry=100.0, stop=98.0, base_height=4.0)
        assert result["risk_per_share"] == 2.0
        assert result["target1_base_height"] == 104.0
        assert result["rr1_base_height"] == 2.0
        assert result["target2_base_height"] == 108.0
        assert result["rr2_base_height"] == 4.0
        assert result["meets_min_reward_risk"] is True

    def test_wide_stop_fails_min_reward_risk(self):
        result = breakout_trade_geometry(entry=100.0, stop=90.0, base_height=5.0)
        assert result["best_reward_risk_ratio"] == 1.0
        assert result["meets_min_reward_risk"] is False

    def test_atr_target_computed_alongside_base_height(self):
        result = breakout_trade_geometry(entry=100.0, stop=99.0, base_height=2.0, atr=1.0)
        assert result["target1_atr"] == 101.0
        assert result["rr1_atr"] == 1.0
        assert result["target2_atr"] == 102.0

    def test_no_target_inputs_returns_no_targets(self):
        result = breakout_trade_geometry(entry=100.0, stop=99.0)
        assert result["best_reward_risk_ratio"] is None
        assert result["meets_min_reward_risk"] is False
        assert "cannot project a target" in result["summary"]

    def test_distant_resistance_leaves_room_to_run(self):
        result = breakout_trade_geometry(entry=100.0, stop=98.0, atr=3.0, overhead_resistance=105.0)
        assert result["rr_at_overhead_resistance"] == 2.5
        assert result["room_to_run"] is True

    def test_close_resistance_caps_reward(self):
        result = breakout_trade_geometry(entry=100.0, stop=98.0, atr=3.0, overhead_resistance=102.0)
        assert result["rr_at_overhead_resistance"] == 1.0
        assert result["room_to_run"] is False
        assert "Do NOT buy into this ceiling" in result["summary"]

    def test_resistance_at_or_below_entry_is_already_cleared(self):
        result = breakout_trade_geometry(entry=100.0, stop=98.0, atr=3.0, overhead_resistance=99.5)
        assert result["room_to_run"] is True
        assert result["rr_at_overhead_resistance"] is None
        assert "already cleared" in result["summary"]

    def test_no_resistance_given_omits_room_to_run(self):
        result = breakout_trade_geometry(entry=100.0, stop=98.0, atr=3.0)
        assert "room_to_run" not in result


def _ohlc(o, h, l, c, v=1000):
    return {"o": o, "h": h, "l": l, "c": c, "v": v}


# A daily series with one clean bullish order block at index 8: a bearish candle
# (the zone 98.5-101.5) followed by a three-bar up impulse that takes out the
# prior swing high (a bullish break of structure), then a return back into the zone.
def _smart_money_daily_bars():
    bars = [_ohlc(100, 100.5, 99.5, 100) for _ in range(8)]
    bars.append(_ohlc(101, 101.5, 98.5, 99))      # 8: bearish order block
    bars.append(_ohlc(99.2, 104.5, 99.0, 104))    # 9: impulse up...
    bars.append(_ohlc(104, 107.5, 103.5, 107))    # 10
    bars.append(_ohlc(107, 110.5, 106.5, 110))    # 11 (breaks structure)
    bars.append(_ohlc(109, 109.5, 105.5, 106))    # 12: return down...
    bars.append(_ohlc(106, 106.5, 102.5, 103))    # 13
    bars.append(_ohlc(103, 103.5, 100.5, 101))    # 14
    bars.append(_ohlc(101, 101.5, 99.5, 100))     # 15: back into the zone
    bars.append(_ohlc(100, 100.8, 99.2, 100))     # 16
    bars.append(_ohlc(100, 100.6, 99.4, 100))     # 17
    return bars


class TestOrderBlocks:
    def test_detects_bullish_order_block_at_displacement(self):
        result = find_order_blocks(_smart_money_daily_bars())
        bull = [b for b in result["order_blocks"] if b["type"] == "bullish"]
        assert len(bull) == 1
        block = bull[0]
        assert block["index"] == 8
        assert block["bottom"] == 98.5
        assert block["top"] == 101.5

    def test_block_marked_mitigated_after_price_returns(self):
        block = [b for b in find_order_blocks(_smart_money_daily_bars())["order_blocks"] if b["type"] == "bullish"][0]
        assert block["mitigated"] is True  # price returned into the zone after the impulse

    def test_not_enough_bars_returns_note(self):
        assert "note" in find_order_blocks(_make_bars([100.0, 101.0, 102.0]))

    def test_analyze_order_blocks_finds_nearest_demand(self):
        result = analyze_order_blocks(_smart_money_daily_bars(), spot=100.0)
        demand = result["nearest_bullish_ob"]
        assert demand is not None
        assert demand["bottom"] == 98.5 and demand["top"] == 101.5


class TestFairValueGaps:
    def test_detects_bullish_gap(self):
        # Middle bar gaps up: bar3 low (102) is above bar1 high (100).
        bars = [_ohlc(99, 100, 98, 99), _ohlc(100, 103, 100, 102.5), _ohlc(103, 104, 102, 103.5)]
        gaps = find_fair_value_gaps(bars)["fair_value_gaps"]
        assert len(gaps) == 1
        assert gaps[0]["type"] == "bullish"
        assert gaps[0]["bottom"] == 100.0 and gaps[0]["top"] == 102.0
        assert gaps[0]["filled"] is False

    def test_gap_marked_filled_when_price_returns(self):
        bars = [
            _ohlc(99, 100, 98, 99),
            _ohlc(100, 103, 100, 102.5),
            _ohlc(103, 104, 102, 103.5),
            _ohlc(103, 103.5, 101, 101.5),  # trades back down into the 100-102 gap
        ]
        gaps = find_fair_value_gaps(bars)["fair_value_gaps"]
        assert gaps[0]["filled"] is True

    def test_analyze_fair_value_gaps_not_enough_bars(self):
        assert "note" in analyze_fair_value_gaps([_ohlc(100, 101, 99, 100)])


class TestSmartMoneyTradeGeometry:
    def test_meets_three_to_one(self):
        result = smart_money_trade_geometry(entry=100.0, stop=98.0, target=107.0)
        assert result["risk_per_share"] == 2.0
        assert result["reward_per_share"] == 7.0
        assert result["reward_risk_ratio"] == 3.5
        assert result["meets_min_reward_risk"] is True

    def test_below_minimum_reward_risk(self):
        result = smart_money_trade_geometry(entry=100.0, stop=98.0, target=104.0)
        assert result["reward_risk_ratio"] == 2.0
        assert result["meets_min_reward_risk"] is False

    def test_invalid_ordering_returns_note(self):
        assert "note" in smart_money_trade_geometry(entry=100.0, stop=101.0, target=105.0)


class TestSmartMoneySetup:
    def test_long_setup_on_confirmed_return_into_demand(self):
        daily = _smart_money_daily_bars()
        # Last intraday bar is a bullish rejection candle (long lower wick, small body).
        intraday = [_ohlc(100, 100.2, 99.8, 100) for _ in range(5)]
        intraday.append(_ohlc(100.0, 100.3, 99.0, 100.2))
        result = analyze_smart_money_setup(daily, intraday_bars=intraday, spot=100.0)
        assert result["signal"] == "long_setup"
        assert result["price_in_order_block"] is True
        assert "rejection_candle" in result["intraday_confirmation"]
        assert result["order_block"]["bottom"] == 98.5
        assert result["meets_min_reward_risk"] is True

    def test_watching_when_price_not_in_block(self):
        daily = _smart_money_daily_bars()
        result = analyze_smart_money_setup(daily, intraday_bars=None, spot=108.0)
        assert result["signal"] == "watching"
        assert result["price_in_order_block"] is False
        assert result["order_block"] is not None

    def test_no_setup_without_enough_daily_bars(self):
        result = analyze_smart_money_setup(_make_bars([100.0, 101.0, 102.0]))
        assert result["signal"] == "no_setup"
        assert "note" in result


class TestPremiumDiscount:
    def test_not_enough_bars_returns_note(self):
        assert "note" in analyze_premium_discount(_make_bars([100.0, 101.0]))

    def test_price_below_midpoint_is_discount(self):
        # Range 90..110 (eq 100); spot 92 sits clearly in the discount half.
        bars = _make_bars([100.0] * 19 + [92.0], highs=[110.0] * 20, lows=[90.0] * 20)
        result = analyze_premium_discount(bars, spot=92.0)
        assert result["zone"] == "discount"
        assert result["in_discount"] is True
        assert result["equilibrium"] == 100.0

    def test_price_above_midpoint_is_premium(self):
        bars = _make_bars([100.0] * 19 + [108.0], highs=[110.0] * 20, lows=[90.0] * 20)
        result = analyze_premium_discount(bars, spot=108.0)
        assert result["zone"] == "premium"
        assert result["in_discount"] is False

    def test_ote_zone_is_deep_discount(self):
        bars = _make_bars([100.0] * 19 + [95.0], highs=[110.0] * 20, lows=[90.0] * 20)
        result = analyze_premium_discount(bars, spot=95.0)
        # OTE = 0.618-0.79 retrace from high: 110 - 0.79*20=94.2 .. 110-0.618*20=97.64
        assert result["ote_zone"]["bottom"] == 94.2
        assert result["in_ote_zone"] is True


class TestLiquidity:
    def test_not_enough_bars_returns_note(self):
        assert "note" in analyze_liquidity(_make_bars([100.0, 101.0]))

    def test_detects_buy_and_sell_side_pools(self):
        # A clean zig-zag creates swing highs (BSL) and swing lows (SSL).
        closes = [100, 104, 100, 104, 100, 104, 100, 104, 100, 102]
        result = analyze_liquidity(_make_bars([float(c) for c in closes]), swing=1, spot=102.0)
        assert result["buy_side_liquidity"]
        assert result["sell_side_liquidity"]

    def test_detects_bullish_sweep_of_sell_side_liquidity(self):
        # Form a swing low at ~99, then a later bar undercuts it and closes back above.
        closes = [105.0, 100.0, 105.0, 104.0, 103.0, 106.0]
        lows = [104.0, 99.0, 104.0, 103.0, 97.0, 105.0]  # bar idx4 pierces the 99 swing low
        highs = [c + 0.5 for c in closes]
        result = analyze_liquidity(_make_bars(closes, highs=highs, lows=lows), swing=1, recent=4, spot=106.0)
        assert result["recent_sweep"] is not None
        assert result["recent_sweep"]["type"] == "bullish"
        assert result["bullish_sweep"] is True

    def test_clusters_equal_highs_into_one_pool(self):
        # Two near-equal swing highs at ~104 should merge into a single equal-highs pool.
        closes = [100, 104, 100, 104.05, 100, 102]
        result = analyze_liquidity(_make_bars([float(c) for c in closes]), swing=1, spot=102.0)
        equal_pools = [p for p in result["buy_side_liquidity"] if p["equal"]]
        assert equal_pools


class TestOpeningRange:
    """analyze_opening_range / compute_opening_range: anchored to the 09:30 ET
    wall clock from bar timestamps, never fabricated from a mid-session start.
    2026-07-16 is EDT (UTC-4), so 09:30 ET = 13:30Z."""

    @staticmethod
    def _bar(ts, h, l, c, v=1000):
        return {"t": ts, "o": c, "h": h, "l": l, "c": c, "v": v}

    def _opening_bars(self):
        # 15 one-minute bars 13:30Z-13:44Z spanning 100.0-102.0.
        return [
            self._bar(f"2026-07-16T13:{30 + i}:00Z", 101.0 + (i == 5), 100.0 + 0.1 * (i == 0), 100.5)
            for i in range(15)
        ]

    def test_range_measured_from_the_bell(self):
        from datetime import datetime, timezone

        from agent_stonks.technical_analysis import analyze_opening_range

        bars = self._opening_bars() + [
            self._bar("2026-07-16T13:50:00Z", 102.5, 101.8, 102.4, v=3000),
            self._bar("2026-07-16T13:51:00Z", 103.0, 102.2, 102.9, v=3000),
        ]
        result = analyze_opening_range(
            bars, minutes=15, now=datetime(2026, 7, 16, 13, 52, tzinfo=timezone.utc)
        )
        assert result["opening_range_high"] == 102.0
        assert result["opening_range_low"] == 100.0
        assert result["status"] == "broken out above"
        assert result["opening_range_date"] == "2026-07-16"
        # Recent 3000-avg bars vs the 1000-avg opening bars.
        assert result["volume_ratio_vs_opening_range"] == 3.0

    def test_mid_session_start_refuses_to_fabricate(self):
        from agent_stonks.technical_analysis import analyze_opening_range

        bars = [  # stream started 12:45 ET -- nowhere near the open
            self._bar("2026-07-16T16:45:00Z", 105.0, 104.0, 104.5),
            self._bar("2026-07-16T16:46:00Z", 105.5, 104.5, 105.0),
        ]
        result = analyze_opening_range(bars, minutes=15)
        assert "opening_range_high" not in result
        assert "refusing to fabricate" in result["note"]

    def test_cached_range_survives_buffer_eviction(self):
        from agent_stonks.technical_analysis import analyze_opening_range

        cached = {
            "date": "2026-07-16",
            "minutes": 15,
            "high": 102.0,
            "low": 100.0,
            "bar_count": 15,
            "avg_volume": 1000.0,
            "complete": True,
        }
        bars = [  # buffer no longer reaches the open
            self._bar("2026-07-16T19:00:00Z", 99.4, 98.9, 99.0),
            self._bar("2026-07-16T19:01:00Z", 99.5, 99.0, 99.2),
        ]
        result = analyze_opening_range(bars, minutes=15, opening_range=cached)
        assert result["opening_range_high"] == 102.0
        assert result["status"] == "broken out below"

    def test_still_forming_inside_the_window(self):
        from datetime import datetime, timezone

        from agent_stonks.technical_analysis import analyze_opening_range

        bars = self._opening_bars()[:5]
        result = analyze_opening_range(
            bars, minutes=15, now=datetime(2026, 7, 16, 13, 36, tzinfo=timezone.utc)
        )
        assert result["status"] == "still forming"

    def test_compute_coverage_grace_and_assume_coverage(self):
        from agent_stonks.technical_analysis import compute_opening_range

        late = [  # earliest bar 13:34Z -- 4 min after the bell, past the 3-min grace
            self._bar(f"2026-07-16T13:{34 + i}:00Z", 101.0, 100.0, 100.5) for i in range(11)
        ]
        assert "high" not in compute_opening_range(late, 15)
        rng = compute_opening_range(late, 15, assume_coverage=True)
        assert rng["high"] == 101.0 and rng["low"] == 100.0

    def test_no_bars_and_unstamped_bars(self):
        from agent_stonks.technical_analysis import analyze_opening_range

        assert "note" in analyze_opening_range([], minutes=15)
        assert "note" in analyze_opening_range([{"h": 1, "l": 1, "c": 1}], minutes=15)

    def test_key_levels_omit_opening_range_for_mid_session_buffer(self):
        # Buffer starts 12:45 ET: session high/low still reported, but no
        # fabricated opening-range levels.
        bars = [
            self._bar("2026-07-16T16:45:00Z", 105.0, 104.0, 104.5),
            self._bar("2026-07-16T16:46:00Z", 105.5, 104.5, 105.0),
        ]
        result = key_levels(bars, daily_bars=None, opening_range_minutes=15)
        assert "session_high" in result["levels"]
        assert "opening_range_high" not in result["levels"]

    def test_key_levels_use_supplied_cached_range(self):
        bars = [
            self._bar("2026-07-16T16:45:00Z", 105.0, 104.0, 104.5),
        ]
        cached = {"date": "2026-07-16", "minutes": 15, "high": 102.0, "low": 100.0,
                  "bar_count": 15, "avg_volume": 1000.0, "complete": True}
        result = key_levels(bars, daily_bars=None, opening_range=cached)
        assert result["levels"]["opening_range_high"] == 102.0
        assert result["levels"]["opening_range_low"] == 100.0


class TestProfile2BracketsSpot:
    def _result(self, spot):
        closes = [100.0] * 15
        closes += [100.0 + 0.1 * i for i in range(1, 16)]
        closes += [closes[-1] - 0.1 * i for i in range(1, 16)]
        vols = [5000] * 15 + [1000] * 30
        vols[30] = 6000
        return analyze_volume_profile_2(_session_bars(closes, vols), spot=spot)

    def test_in_range_spot_brackets(self):
        result = self._result(spot=100.5)
        assert result["brackets_spot"] is True
        assert result["warning"] is None

    def test_out_of_range_spot_warns_loudly(self):
        # A spot several percent outside the session's range means the map is
        # describing some other tape (wrong day, stale fetch) -- the result
        # must say so instead of returning a normal-looking level list.
        result = self._result(spot=110.0)
        assert result["brackets_spot"] is False
        assert "WARNING" in result["warning"]
        assert "WARNING" in result["summary"]

    def test_slightly_beyond_range_is_tolerated(self):
        # ~15-min delayed source bars: a genuine breakout just past the mapped
        # high must not read as a corrupt map.
        top = self._result(spot=100.5)["range_high"]
        result = self._result(spot=top * 1.005)
        assert result["brackets_spot"] is True
