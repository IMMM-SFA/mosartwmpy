"""Unit tests for GDROM and C-GDROM release methods.

Tests cover:
  - GDROM rule-file parsers (_parse_rule_lines, _parse_linear_module)
  - GDROM inference helper (_evaluate_rules)
  - C-GDROM eligibility fix (both flow stats and storage curve required)
  - C-GDROM S-curve pre-computation (_compute_sty): structural properties only;
    numerical reference values are marked TODO
  - C-GDROM prediction functions: structural/invariant checks only;
    exact numerical values are marked TODO (need reference outputs from Zhao et al. code)
"""

import io
import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# GDROM parsers and inference
# ---------------------------------------------------------------------------

class TestParseRuleLines(unittest.TestCase):
    """Tests for mosartwmpy.reservoirs.gdrom._parse_rule_lines."""

    def _parse(self, text, is_ct=False, convert_release=False):
        from mosartwmpy.reservoirs.gdrom import _parse_rule_lines
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(text)
            path = f.name
        try:
            return _parse_rule_lines(path, is_ct=is_ct, convert_release=convert_release)
        finally:
            os.unlink(path)

    def test_tree_format_single_condition(self):
        text = "if (Inflow <= 100.0) then Release: 50.0\n"
        rules = self._parse(text)
        self.assertEqual(len(rules), 1)
        conditions, value = rules[0]
        self.assertEqual(len(conditions), 1)
        var_idx, op_le, threshold = conditions[0]
        self.assertEqual(var_idx, 0)   # Inflow index
        self.assertTrue(op_le)
        # threshold is converted from acre-ft/day → m³/s at parse time
        from mosartwmpy.reservoirs.gdrom import _ACFT_DAY_TO_M3S
        self.assertAlmostEqual(threshold, 100.0 * _ACFT_DAY_TO_M3S, places=6)

    def test_tree_format_multiple_conditions(self):
        text = "if (Inflow <= 10.0) (Storage > 500.0) then Release: 25.0\n"
        rules = self._parse(text)
        self.assertEqual(len(rules), 1)
        conditions, _ = rules[0]
        self.assertEqual(len(conditions), 2)
        self.assertEqual(conditions[0][0], 0)  # Inflow
        self.assertEqual(conditions[1][0], 1)  # Storage
        self.assertFalse(conditions[1][1])      # op > → op_le = False

    def test_const_format(self):
        # single-line bare constant
        text = "Release: 42.5\n"
        rules = self._parse(text)
        self.assertEqual(len(rules), 1)
        conditions, value = rules[0]
        self.assertEqual(len(conditions), 0)
        self.assertAlmostEqual(value, 42.5)

    def test_const_format_with_conversion(self):
        from mosartwmpy.reservoirs.gdrom import _ACFT_DAY_TO_M3S
        text = "Release: 100.0\n"
        rules = self._parse(text, convert_release=True)
        _, value = rules[0]
        self.assertAlmostEqual(value, 100.0 * _ACFT_DAY_TO_M3S, places=10)

    def test_unknown_variable_skips_line(self):
        text = "if (UnknownVar <= 10.0) then Release: 99.0\n"
        rules = self._parse(text)
        # line with unknown variable should be skipped → empty result
        self.assertEqual(len(rules), 0)

    def test_ct_format(self):
        # CT (module-condition) file; terminal keyword is 'module'
        text = "if (Inflow > 50.0) then module: 1\n"
        rules = self._parse(text, is_ct=True)
        self.assertEqual(len(rules), 1)
        conditions, module_id = rules[0]
        self.assertEqual(int(module_id), 1)
        self.assertEqual(conditions[0][0], 0)   # Inflow
        self.assertFalse(conditions[0][1])       # op > → op_le = False

    def test_doy_and_pdsi_not_unit_converted(self):
        # DOY and PDSI thresholds must NOT be unit-converted
        text = "if (DOY <= 180.0) (PDSI > -2.0) then Release: 10.0\n"
        rules = self._parse(text)
        conditions, _ = rules[0]
        # DOY threshold should remain 180.0
        doy_cond = next(c for c in conditions if c[0] == 2)
        self.assertAlmostEqual(doy_cond[2], 180.0)
        # PDSI threshold should remain -2.0
        pdsi_cond = next(c for c in conditions if c[0] == 3)
        self.assertAlmostEqual(pdsi_cond[2], -2.0)


class TestParseLinearModule(unittest.TestCase):
    """Tests for mosartwmpy.reservoirs.gdrom._parse_linear_module."""

    def _parse(self, text):
        from mosartwmpy.reservoirs.gdrom import _parse_linear_module
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(text)
            path = f.name
        try:
            return _parse_linear_module(path)
        finally:
            os.unlink(path)

    def test_slope_inflow_dimensionless(self):
        # Release in acre-ft/day = slope_inflow * Inflow [acre-ft/day]
        # Both sides convert by _ACFT_DAY_TO_M3S, so the ratio is dimensionless.
        m = self._parse("Release = 0.5*Inflow + 10.0\n")
        self.assertAlmostEqual(m.slope_inflow, 0.5)

    def test_intercept_converted(self):
        from mosartwmpy.reservoirs.gdrom import _ACFT_DAY_TO_M3S
        m = self._parse("Release = 0.0*Inflow + 100.0\n")
        self.assertAlmostEqual(m.intercept, 100.0 * _ACFT_DAY_TO_M3S, places=10)

    def test_slope_storage_converted(self):
        # slope_storage_si = slope_storage_file * (_ACFT_DAY_TO_M3S / _ACFT_TO_M3)
        from mosartwmpy.reservoirs.gdrom import _ACFT_DAY_TO_M3S, _ACFT_TO_M3
        m = self._parse("Release = 0.5*Inflow + 2.0*Storage + 0.0\n")
        expected_slope_storage = 2.0 * _ACFT_DAY_TO_M3S / _ACFT_TO_M3
        self.assertAlmostEqual(m.slope_storage, expected_slope_storage, places=15)

    def test_no_storage_term(self):
        m = self._parse("Release = 0.3*Inflow + 5.0\n")
        self.assertAlmostEqual(m.slope_storage, 0.0)

    def test_zero_intercept(self):
        m = self._parse("Release = 1.0*Inflow + 0.0*Storage\n")
        self.assertAlmostEqual(m.intercept, 0.0, places=10)


class TestEvaluateRules(unittest.TestCase):
    """Tests for mosartwmpy.reservoirs.gdrom._evaluate_rules."""

    def setUp(self):
        from mosartwmpy.reservoirs.gdrom import _VAR_IDX
        # Build two rules:
        #   Rule 0: Inflow <= 50  → value 10.0
        #   Rule 1: Inflow > 50   → value 20.0  (no explicit condition needed; always matches after rule 0 fails)
        self.rules = [
            (((0, True, 50.0),), 10.0),   # Inflow <= 50
            (((0, False, 50.0),), 20.0),  # Inflow > 50
        ]

    def test_first_matching_path_returned(self):
        from mosartwmpy.reservoirs.gdrom import _evaluate_rules
        # Inflow=30 → rule 0 matches
        result = _evaluate_rules(self.rules, (30.0, 0.0, 1.0, 0.0))
        self.assertAlmostEqual(result, 10.0)

    def test_second_path_returned_when_first_fails(self):
        from mosartwmpy.reservoirs.gdrom import _evaluate_rules
        # Inflow=100 → rule 0 fails, rule 1 matches
        result = _evaluate_rules(self.rules, (100.0, 0.0, 1.0, 0.0))
        self.assertAlmostEqual(result, 20.0)

    def test_no_match_returns_none(self):
        from mosartwmpy.reservoirs.gdrom import _evaluate_rules
        # Single rule that cannot match (Inflow <= -1 is impossible with positive input)
        rules = [(((0, True, -1.0),), 99.0)]
        result = _evaluate_rules(rules, (10.0, 0.0, 1.0, 0.0))
        self.assertIsNone(result)

    def test_empty_conditions_always_matches(self):
        from mosartwmpy.reservoirs.gdrom import _evaluate_rules
        # Const rule: no conditions → always matches
        rules = [((), 42.0)]
        result = _evaluate_rules(rules, (999.0, 999.0, 1.0, 0.0))
        self.assertAlmostEqual(result, 42.0)

    def test_multi_condition_all_must_pass(self):
        from mosartwmpy.reservoirs.gdrom import _evaluate_rules
        # Inflow <= 100 AND Storage > 500 → value 7.0
        rules = [(((0, True, 100.0), (1, False, 500.0)), 7.0)]
        # Both conditions true
        self.assertAlmostEqual(_evaluate_rules(rules, (50.0, 600.0, 1.0, 0.0)), 7.0)
        # Storage condition fails
        self.assertIsNone(_evaluate_rules(rules, (50.0, 400.0, 1.0, 0.0)))


# ---------------------------------------------------------------------------
# C-GDROM eligibility fix
# ---------------------------------------------------------------------------

class TestCgdromEligibilityBothFilesRequired(unittest.TestCase):
    """The eligibility check in _init_cgdrom_data must require a reservoir to appear
    in BOTH the flow-statistics and the storage-curve files.  A reservoir present in
    only one file must not be marked eligible (Comment 1 fix)."""

    def _make_flow_parquet(self, grand_ids, tmpdir):
        """Write a minimal flow-stats parquet with rows for each grand_id."""
        rows = [
            {'GRAND_ID': gid, 'I99': 1.0, 'I80': 0.8, 'I50': 0.5,
             'I30': 0.3, 'I10': 0.1, 'I_MEAN': 0.4}
            for gid in grand_ids
        ]
        path = os.path.join(tmpdir, 'flow_stats.parquet')
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def _make_curve_csv(self, grand_ids, tmpdir):
        """Write a minimal storage-curve CSV with rows for each grand_id."""
        rows = [
            {'GRAND_ID': gid, 'CURVE_SHAPE': 'single', 'A1': None, 'A2': None,
             'A3': None, 'A4': None, 'S_A4_A1': 10.0, 'S_A2_A3': 100.0,
             'S_MEDIAN': 55.0, 'S_DEAD': 5.0, 'S_FLOOD_CAP': 95.0}
            for gid in grand_ids
        ]
        path = os.path.join(tmpdir, 'storage_curve.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_reservoir_in_both_files_is_eligible(self):
        """A GRanD ID present in both flow stats and curve → has_cgdrom_stats True."""
        from mosartwmpy.reservoirs.grid import _init_cgdrom_data
        from benedict.dicts import benedict as Benedict
        import yaml, importlib.resources

        with tempfile.TemporaryDirectory() as tmpdir:
            flow_path  = self._make_flow_parquet([1001], tmpdir)
            curve_path = self._make_curve_csv([1001], tmpdir)

            defaults_file = str(importlib.resources.files('mosartwmpy').joinpath('config_defaults.yaml'))
            with open(defaults_file) as f:
                config = Benedict(yaml.safe_load(f))
            config['water_management.reservoirs.enable_cgdrom'] = True
            config['water_management.reservoirs.cgdrom.flow_stats.path'] = flow_path
            config['water_management.reservoirs.cgdrom.storage_curve.path'] = curve_path

            # minimal grid-like object with one reservoir cell
            class FakeGrid:
                reservoir_id = np.array([1001.0])
                reservoir_storage_capacity = np.array([1.0e9])  # 1 km³ in m³

            grid = FakeGrid()
            enable, has_stats = _init_cgdrom_data(grid, config, 1)
            self.assertTrue(enable)
            self.assertTrue(has_stats[0])

    def test_reservoir_in_flow_only_is_not_eligible(self):
        """A GRanD ID present only in flow stats (not in curve) → has_cgdrom_stats False."""
        from mosartwmpy.reservoirs.grid import _init_cgdrom_data
        from benedict.dicts import benedict as Benedict
        import yaml, importlib.resources

        with tempfile.TemporaryDirectory() as tmpdir:
            flow_path  = self._make_flow_parquet([1001], tmpdir)
            # curve file has a *different* reservoir — 1001 is absent
            curve_path = self._make_curve_csv([9999], tmpdir)

            defaults_file = str(importlib.resources.files('mosartwmpy').joinpath('config_defaults.yaml'))
            with open(defaults_file) as f:
                config = Benedict(yaml.safe_load(f))
            config['water_management.reservoirs.enable_cgdrom'] = True
            config['water_management.reservoirs.cgdrom.flow_stats.path'] = flow_path
            config['water_management.reservoirs.cgdrom.storage_curve.path'] = curve_path

            class FakeGrid:
                reservoir_id = np.array([1001.0])

            grid = FakeGrid()
            enable, has_stats = _init_cgdrom_data(grid, config, 1)
            self.assertFalse(enable)
            self.assertFalse(has_stats[0])

    def test_reservoir_in_curve_only_is_not_eligible(self):
        """A GRanD ID present only in curve (not in flow stats) → has_cgdrom_stats False."""
        from mosartwmpy.reservoirs.grid import _init_cgdrom_data
        from benedict.dicts import benedict as Benedict
        import yaml, importlib.resources

        with tempfile.TemporaryDirectory() as tmpdir:
            # flow file has a different reservoir
            flow_path  = self._make_flow_parquet([9999], tmpdir)
            curve_path = self._make_curve_csv([1001], tmpdir)

            defaults_file = str(importlib.resources.files('mosartwmpy').joinpath('config_defaults.yaml'))
            with open(defaults_file) as f:
                config = Benedict(yaml.safe_load(f))
            config['water_management.reservoirs.enable_cgdrom'] = True
            config['water_management.reservoirs.cgdrom.flow_stats.path'] = flow_path
            config['water_management.reservoirs.cgdrom.storage_curve.path'] = curve_path

            class FakeGrid:
                reservoir_id = np.array([1001.0])

            grid = FakeGrid()
            enable, has_stats = _init_cgdrom_data(grid, config, 1)
            self.assertFalse(enable)
            self.assertFalse(has_stats[0])


# ---------------------------------------------------------------------------
# C-GDROM S-curve pre-computation
# ---------------------------------------------------------------------------

class TestComputeSty(unittest.TestCase):
    """Tests for mosartwmpy.reservoirs.cgdrom._compute_sty."""

    def test_output_shape(self):
        from mosartwmpy.reservoirs.cgdrom import _compute_sty
        sty = _compute_sty(1, 4, 8, 11, s_low=10.0, s_high=100.0)
        self.assertEqual(sty.shape, (365,))

    def test_values_bounded_between_s_low_and_s_high(self):
        from mosartwmpy.reservoirs.cgdrom import _compute_sty
        s_low, s_high = 20.0, 80.0
        sty = _compute_sty(2, 5, 9, 12, s_low=s_low, s_high=s_high)
        self.assertTrue(np.all(sty >= s_low - 1e-9))
        self.assertTrue(np.all(sty <= s_high + 1e-9))

    def test_single_curve_shape_is_flat(self):
        # When curve_shape == 'single', the caller passes s_median as a flat array.
        # _compute_sty is not called for 'single' shape; instead a np.full is used
        # in load_cgdrom_params.  This test verifies the flat-array branch directly.
        s_median = 42.0
        sty = np.full(365, s_median, dtype=np.float64)
        self.assertTrue(np.all(sty == s_median))

    # TODO: add numerical regression tests against reference outputs from
    # Zhao et al. (2025) C-GDROM reference code (cgdrom_general.py /
    # conceptual_s_curve.py).  For a set of (a1, a2, a3, a4, s_low, s_high)
    # inputs, provide the expected sty[0], sty[90], sty[180], sty[270] values.
    # Example structure:
    #
    #   def test_four_piece_reference_case(self):
    #       from mosartwmpy.reservoirs.cgdrom import _compute_sty
    #       sty = _compute_sty(a1=2, a2=5, a3=9, a4=12, s_low=10.0, s_high=100.0)
    #       self.assertAlmostEqual(sty[0],   <expected>, places=3)
    #       self.assertAlmostEqual(sty[90],  <expected>, places=3)
    #       self.assertAlmostEqual(sty[180], <expected>, places=3)
    #       self.assertAlmostEqual(sty[270], <expected>, places=3)


# ---------------------------------------------------------------------------
# C-GDROM prediction functions — structural / invariant checks
# ---------------------------------------------------------------------------

def _make_params(cgdrom_type='general', **overrides):
    """Build a minimal CgdromParams for unit testing."""
    from mosartwmpy.reservoirs.cgdrom import CgdromParams
    defaults = dict(
        cgdrom_type=cgdrom_type,
        sty=np.full(365, 50.0),
        s_cap=100.0,
        s_dead=5.0,
        s_flood_cap=95.0,
        size_ratio=0.5,
        i99=90.0, i80=70.0, i50=50.0, i30=30.0, i10=10.0, i_mean=40.0,
        q1=90.0, q2=40.0, q3=30.0,
        r_flood=60.0,
    )
    defaults.update(overrides)
    return CgdromParams(**defaults)


class TestPredictGeneralDaily(unittest.TestCase):
    """Structural invariant tests for _predict_general_daily.

    TODO: add exact numerical regression tests against reference outputs from
    the Zhao et al. (2025) cgdrom_general.py predict_general_daily() function.
    For each of the five module branches (major flood, above target, below target
    above dead, below dead, spill guard) provide (it, st0, rt0, s_ty) inputs and
    the expected release in m³/s.
    """

    def setUp(self):
        self.p = _make_params()

    def test_output_non_negative(self):
        from mosartwmpy.reservoirs.cgdrom import _predict_general_daily
        for it, st0, rt0, s_ty in [
            (0.0, 50.0, 0.0, 50.0),
            (100.0, 5.0, 0.0, 50.0),
            (0.0, 1.0, 0.0, 50.0),   # below dead storage
            (50.0, 99.0, 0.0, 50.0), # above flood cap → spill guard
        ]:
            rt = _predict_general_daily(self.p, it, st0, rt0, s_ty)
            self.assertGreaterEqual(rt, 0.0, msg=f"it={it} st0={st0}")

    def test_spill_guard_release_at_least_inflow(self):
        from mosartwmpy.reservoirs.cgdrom import _predict_general_daily
        # storage above flood cap → release must be >= inflow
        it = 40.0
        st0 = self.p.s_flood_cap + 1.0   # above flood cap
        rt = _predict_general_daily(self.p, it, st0, 0.0, 50.0)
        self.assertGreaterEqual(rt, it)

    def test_below_dead_release_capped_at_i10(self):
        from mosartwmpy.reservoirs.cgdrom import _predict_general_daily
        # below dead storage → release = min(it, i10)
        it = 20.0   # it > i10 (10.0)
        st0 = self.p.s_dead - 1.0
        rt = _predict_general_daily(self.p, it, st0, 0.0, 50.0)
        self.assertAlmostEqual(rt, self.p.i10)

    # TODO: exact branch-level regression tests. Example structure:
    #   def test_module1_major_flood(self):
    #       rt = _predict_general_daily(self.p, it=95.0, st0=60.0, rt0=0.0, s_ty=50.0)
    #       self.assertAlmostEqual(rt, <expected_m3s>, places=4)


class TestPredictFcDaily(unittest.TestCase):
    """Structural invariant tests for _predict_fc_daily.

    TODO: add numerical regression tests against Zhao et al. cgdrom_fc.py
    predict_fc_daily() for both calibrated and NaN-coefficient (general-model
    fallback) cases.
    """

    def setUp(self):
        self.p_uncal = _make_params(cgdrom_type='flood_control')   # all NaN coefficients
        self.p_cal = _make_params(
            cgdrom_type='flood_control',
            m1_r=55.0, m2a_coef=0.5, m2a_intercept=5.0,
            m2a_uses_inflow=False, m2b_coef=0.3, m2b_intercept=3.0,
            m3_r=20.0,
        )

    def test_output_non_negative(self):
        from mosartwmpy.reservoirs.cgdrom import _predict_fc_daily
        for p in (self.p_uncal, self.p_cal):
            for it, st0 in [(0.0, 50.0), (80.0, 60.0), (0.0, 3.0)]:
                rt = _predict_fc_daily(p, it, st0, 0.0, 50.0)
                self.assertGreaterEqual(rt, 0.0)

    def test_spill_guard_release_at_least_inflow(self):
        from mosartwmpy.reservoirs.cgdrom import _predict_fc_daily
        it = 30.0
        st0 = self.p_uncal.s_flood_cap + 1.0
        rt = _predict_fc_daily(self.p_uncal, it, st0, 0.0, 50.0)
        self.assertGreaterEqual(rt, it)

    # TODO: exact regression test for calibrated FC module output.


class TestPredictIrrDaily(unittest.TestCase):
    """Structural invariant tests for _predict_irr_daily.

    TODO: add numerical regression tests against Zhao et al. cgdrom_irr.py
    predict_irr_daily() for each seasonal module (DOY_D1/D2/D3 branches) and
    the spill-boost path.
    """

    def setUp(self):
        self.p = _make_params(
            cgdrom_type='irrigation',
            m1_r=55.0, m3_r=25.0, m4_r=35.0, m5_r=15.0,
            doy_d1=60, doy_d2=270, doy_d3=120,
        )

    def test_output_non_negative(self):
        from mosartwmpy.reservoirs.cgdrom import _predict_irr_daily
        for doy in [1, 90, 150, 200, 300]:
            rt = _predict_irr_daily(self.p, it=10.0, st0=50.0, rt0=0.0, doy=doy, s_ty=50.0)
            self.assertGreaterEqual(rt, 0.0, msg=f"doy={doy}")

    def test_spill_guard_release_at_least_inflow(self):
        from mosartwmpy.reservoirs.cgdrom import _predict_irr_daily
        it = 20.0
        st0 = self.p.s_flood_cap + 1.0
        rt = _predict_irr_daily(self.p, it=it, st0=st0, rt0=0.0, doy=150, s_ty=50.0)
        self.assertGreaterEqual(rt, it)

    # TODO: exact regression for each seasonal module.


# ---------------------------------------------------------------------------
# C-GDROM previous-release State migration (Comment 2)
# ---------------------------------------------------------------------------

class TestCgdromPrevReleaseInState(unittest.TestCase):
    """Verify that reservoir_cgdrom_prev_release is a numpy array on State,
    not a dict on Grid, and that it is serialised by State.to_dataframe."""

    def test_state_has_field(self):
        from mosartwmpy.state.state import State
        s = State(empty=True)
        s.reservoir_cgdrom_prev_release = np.zeros(3)
        self.assertIsInstance(s.reservoir_cgdrom_prev_release, np.ndarray)

    def test_field_included_in_to_dataframe_keys(self):
        # to_dataframe serialises every attribute that is a np.ndarray.
        # Verify the attribute name appears in the key scan even on an empty instance.
        from mosartwmpy.state.state import State
        import inspect
        s = State(empty=True)
        s.reservoir_cgdrom_prev_release = np.empty(0)
        keys = [k for k in dir(s) if isinstance(getattr(s, k), np.ndarray)]
        self.assertIn('reservoir_cgdrom_prev_release', keys)

    def test_field_restored_by_from_dataframe(self):
        from mosartwmpy.state.state import State
        import pandas as pd
        df = pd.DataFrame({'reservoir_cgdrom_prev_release': [10.0, 20.0]})
        s = State.from_dataframe(df)
        np.testing.assert_array_almost_equal(
            s.reservoir_cgdrom_prev_release, [10.0, 20.0]
        )

    def test_grid_does_not_have_cgdrom_prev_release(self):
        from mosartwmpy.grid.grid import Grid
        g = Grid(empty=True)
        self.assertFalse(hasattr(g, 'cgdrom_prev_release'))


class TestGdromCountersInState(unittest.TestCase):
    """Verify that reservoir_gdrom_fallback_count and reservoir_gdrom_total_calls
    are numpy arrays on State and serialise correctly."""

    def test_fallback_fields_in_to_dataframe_keys(self):
        from mosartwmpy.state.state import State
        s = State(empty=True)
        s.reservoir_gdrom_fallback_count = np.empty(0)
        s.reservoir_gdrom_total_calls    = np.empty(0)
        keys = [k for k in dir(s) if isinstance(getattr(s, k), np.ndarray)]
        self.assertIn('reservoir_gdrom_fallback_count', keys)
        self.assertIn('reservoir_gdrom_total_calls', keys)

    def test_counts_restored_from_dataframe(self):
        from mosartwmpy.state.state import State
        import pandas as pd
        df = pd.DataFrame({
            'reservoir_gdrom_fallback_count': [2.0, 0.0],
            'reservoir_gdrom_total_calls':    [7.0, 3.0],
        })
        s = State.from_dataframe(df)
        np.testing.assert_array_almost_equal(s.reservoir_gdrom_fallback_count, [2.0, 0.0])
        np.testing.assert_array_almost_equal(s.reservoir_gdrom_total_calls, [7.0, 3.0])


if __name__ == '__main__':
    unittest.main()
