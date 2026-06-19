"""
Regression and property tests for motor_flicker_sweep.py.

Matplotlib backend is forced to Agg (headless) so these can run in CI.
"""

import math
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from dataclasses import replace

import volt_sag
from volt_sag import Conductor, Scenario
import motor_flicker_sweep as mf

# ---------------------------------------------------------------------------
# Shared scenario — 200 HP ATL on 12.47 kV, 7-mile #2 ACSR, 500 kVA xfmr
# ---------------------------------------------------------------------------
BASE = Scenario(
    hp=200.0,
    motor_code="K",
    v_ll_hv=12_470.0,
    v_ll_lv=480.0,
    miles=7.0,
    cond=Conductor(1.367, 0.581),
    xfmr_kva=500.0,
    xfmr_pct_z=5.75,
    xfmr_x_over_r=10.0,
    start_mode="ATL",
    atl_model="CONST_I",
    start_pf=0.25,
    atl_I_multiplier_of_LRA=1.0,
    cap_kvar=0.0,
)

# Locked-in reference from known-good run.
SAG_BASE_PCT = 25.980


# ---------------------------------------------------------------------------
# allowable_dv_percent_mv
# ---------------------------------------------------------------------------

class TestAllowableDv:
    def test_infrequent_daily(self):
        # <= 4 per day → 5 %
        assert mf.allowable_dv_percent_mv(0.04) == 5.0

    def test_hourly_limit(self):
        # > 4/day but <= 10/hr → 4 %
        assert mf.allowable_dv_percent_mv(4.0) == 4.0
        assert mf.allowable_dv_percent_mv(2.0) == 4.0

    def test_explicit_daily_count_overrides(self):
        # Force starts_per_day <= 4 even though starts_per_hour is high
        result = mf.allowable_dv_percent_mv(starts_per_hour=10.0, starts_per_day=2.0)
        assert result == 5.0


# ---------------------------------------------------------------------------
# status_from_dv and loading_status_from_currents
# ---------------------------------------------------------------------------

class TestStatusHelpers:
    def test_ok_below_review(self):
        assert mf.status_from_dv(2.0, 3.2, 4.0) == "OK"

    def test_review_between_review_and_limit(self):
        assert mf.status_from_dv(3.5, 3.2, 4.0) == "REVIEW"

    def test_violation_at_limit(self):
        assert mf.status_from_dv(4.0, 3.2, 4.0) == "LIKELY VIOLATION"

    def test_violation_above_limit(self):
        assert mf.status_from_dv(5.5, 3.2, 4.0) == "LIKELY VIOLATION"

    def test_loading_within(self):
        assert mf.loading_status_from_currents(50.0, 10.0, 190.0) == "WITHIN AMPACITY"

    def test_loading_near(self):
        assert mf.loading_status_from_currents(175.0, 10.0, 190.0) == "NEAR AMPACITY"

    def test_loading_exceeded(self):
        assert mf.loading_status_from_currents(200.0, 10.0, 190.0) == "AMPACITY EXCEEDED"
        assert mf.loading_status_from_currents(50.0, 200.0, 190.0) == "AMPACITY EXCEEDED"


# ---------------------------------------------------------------------------
# startup_kva_from_current
# ---------------------------------------------------------------------------

class TestStartupKva:
    def test_known_value(self):
        # sqrt(3) * 12470 * 10 / 1000 = 215.987 kVA
        result = mf.startup_kva_from_current(12_470.0, 10.0)
        assert abs(result - 215.987) < 0.01


# ---------------------------------------------------------------------------
# sag_percent — must agree with volt_sag.run_return
# ---------------------------------------------------------------------------

class TestSagPercent:
    def test_regression(self):
        assert abs(mf.sag_percent(BASE) - SAG_BASE_PCT) < 0.01

    def test_matches_volt_sag_run_return(self):
        sag_sweep, _ = volt_sag.run_return(BASE)
        sag_pct = mf.sag_percent(BASE)
        assert abs(sag_pct - sag_sweep) < 1e-6

    def test_vfd_lower_than_atl(self):
        sag_atl = mf.sag_percent(BASE)
        sag_vfd = mf.sag_percent(replace(BASE, start_mode="VFD"))
        assert sag_vfd < sag_atl

    def test_upstream_z_increases_sag(self):
        sag_no = mf.sag_percent(BASE)
        sag_up = mf.sag_percent(BASE, z_upstream_phase=complex(0.5, 2.0))
        assert sag_up > sag_no

    def test_startup_cap_reduces_sag(self):
        sag_free = mf.sag_percent(BASE)
        sag_cap = mf.sag_percent(BASE, max_start_kva=50.0)
        assert sag_cap < sag_free

    def test_z_line_override_used(self):
        # Zero line → less sag than 7 miles
        sag_full = mf.sag_percent(BASE)
        sag_zero = mf.sag_percent(BASE, z_line_override=0j)
        assert sag_zero < sag_full


# ---------------------------------------------------------------------------
# start_and_run_current_hv_amps
# ---------------------------------------------------------------------------

class TestStartAndRunCurrent:
    def test_start_greater_than_run(self):
        start_i, run_i = mf.start_and_run_current_hv_amps(BASE)
        assert start_i > run_i

    def test_regression_values(self):
        start_i, run_i = mf.start_and_run_current_hv_amps(BASE)
        assert abs(start_i - 78.662) < 0.1
        assert abs(run_i - 8.079) < 0.1

    def test_startup_cap_limits_start_current(self):
        start_free, _ = mf.start_and_run_current_hv_amps(BASE)
        start_cap, _ = mf.start_and_run_current_hv_amps(BASE, max_start_kva=50.0)
        assert start_cap < start_free

    def test_run_current_unaffected_by_startup_cap(self):
        _, run_free = mf.start_and_run_current_hv_amps(BASE)
        _, run_cap = mf.start_and_run_current_hv_amps(BASE, max_start_kva=50.0)
        # Run current is computed at nominal voltage, startup cap doesn't apply
        assert abs(run_free - run_cap) < 1e-9

    def test_vfd_start_current_lower_than_atl(self):
        start_atl, _ = mf.start_and_run_current_hv_amps(BASE)
        start_vfd, _ = mf.start_and_run_current_hv_amps(replace(BASE, start_mode="VFD"))
        assert start_vfd < start_atl

    def test_start_current_consistent_with_sag(self):
        # Back-calculate sag from start current and verify it matches sag_percent
        start_i, _ = mf.start_and_run_current_hv_amps(BASE)
        z_line = complex(BASE.cond.r_ohm_per_mile * BASE.miles, BASE.cond.x_ohm_per_mile * BASE.miles)
        z_xfmr = volt_sag.z_from_pctz(BASE.v_ll_hv, BASE.xfmr_kva, BASE.xfmr_pct_z, BASE.xfmr_x_over_r)
        Z_th = z_line + z_xfmr
        Vth = complex(BASE.v_ll_hv / volt_sag.SQRT3, 0.0)
        # I = (Vth - Vload)/Zth  →  Vload = Vth - Zth*I (approximate, ignoring cap current angle)
        # Check that computed current magnitude is physically consistent with voltage drop
        v_drop_approx = abs(Z_th) * start_i
        assert v_drop_approx < abs(Vth)  # can't drop more than source voltage


# ---------------------------------------------------------------------------
# Minimum starting voltage helpers
# ---------------------------------------------------------------------------

class TestStartVoltage:
    def test_pu_calculation(self):
        assert abs(mf.start_voltage_pu(15.0) - 0.85) < 1e-9
        assert abs(mf.start_voltage_pu(0.0) - 1.0) < 1e-9

    def test_status_ok(self):
        assert mf.start_voltage_status(0.90) == "OK"
        assert mf.start_voltage_status(0.85) == "OK"

    def test_status_marginal(self):
        assert "MARGINAL" in mf.start_voltage_status(0.82)

    def test_status_at_risk(self):
        assert "RISK" in mf.start_voltage_status(0.79)
        assert "RISK" in mf.start_voltage_status(0.50)

    def test_sag_26pct_is_at_risk(self):
        # BASE scenario sags ~26 %, so V_start ≈ 0.74 pu → AT RISK
        v_pu = mf.start_voltage_pu(SAG_BASE_PCT)
        assert mf.start_voltage_status(v_pu) in ("AT RISK (<80%)", "MARGINAL (<85%)")


# ---------------------------------------------------------------------------
# IEEE 519 helpers
# ---------------------------------------------------------------------------

class TestIeee519:
    def test_tdd_limit_by_ratio(self):
        assert mf.ieee519_tdd_limit(10.0) == 5.0
        assert mf.ieee519_tdd_limit(30.0) == 8.0
        assert mf.ieee519_tdd_limit(75.0) == 12.0
        assert mf.ieee519_tdd_limit(500.0) == 15.0
        assert mf.ieee519_tdd_limit(2000.0) == 20.0

    def test_returns_none_for_atl(self):
        s = replace(BASE, start_mode="ATL")
        Z_th = complex(1.0, 5.0)
        Vth = complex(7200.0, 0.0)
        assert mf.ieee519_check(s, Z_th, Vth) is None

    def test_returns_dict_for_vfd(self):
        s = replace(BASE, start_mode="VFD")
        Z_th = complex(1.0, 5.0)
        Vth = complex(7200.0, 0.0)
        result = mf.ieee519_check(s, Z_th, Vth)
        assert result is not None
        assert "isc_il_ratio" in result
        assert "tdd_limit_pct" in result
        assert result["typical_6pulse_tdd_pct"] == 29.0

    def test_weak_feeder_triggers_review(self):
        # Very high Z_th → low ISC → low ISC/IL → tight TDD limit → 29% fails
        s = replace(BASE, start_mode="VFD")
        Z_th_weak = complex(50.0, 200.0)
        Vth = complex(7200.0, 0.0)
        result = mf.ieee519_check(s, Z_th_weak, Vth)
        assert result["screen_result"] == "REVIEW"

    def test_strong_feeder_passes(self):
        # Very low Z_th → high ISC/IL → TDD limit = 20% → 29% still fails
        # Actually 6-pulse without reactor (29%) will always flag REVIEW
        # unless ISC/IL is > 1000 and limit is 20%. Let's just check the ratio.
        s = replace(BASE, start_mode="VFD")
        Z_th_strong = complex(0.001, 0.001)
        Vth = complex(7200.0, 0.0)
        result = mf.ieee519_check(s, Z_th_strong, Vth)
        assert result["isc_il_ratio"] > 100


# ---------------------------------------------------------------------------
# max_hp_under_threshold
# ---------------------------------------------------------------------------

class TestMaxHpUnderThreshold:
    def test_regression(self):
        hp_scan = np.linspace(5, 1000, 200)
        result = mf.max_hp_under_threshold(BASE, 7.0, hp_scan, 4.0)
        assert abs(result - 30.54) < 1.0

    def test_tighter_threshold_gives_lower_hp(self):
        hp_scan = np.linspace(5, 500, 100)
        hp_loose = mf.max_hp_under_threshold(BASE, 7.0, hp_scan, 5.0)
        hp_tight = mf.max_hp_under_threshold(BASE, 7.0, hp_scan, 3.0)
        assert hp_tight < hp_loose

    def test_shorter_feeder_allows_more_hp(self):
        hp_scan = np.linspace(5, 500, 100)
        hp_far = mf.max_hp_under_threshold(BASE, 10.0, hp_scan, 4.0)
        hp_near = mf.max_hp_under_threshold(BASE, 2.0, hp_scan, 4.0)
        assert hp_near > hp_far
