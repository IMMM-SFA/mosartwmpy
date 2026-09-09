"""C-GDROM (Conceptual + Generic Data-driven Reservoir Operation Model) release method.

Implements three per-reservoir operation variants keyed to data availability:
  - General      — no calibration; empirical S-curve + inflow percentile thresholds.
  - Flood control — calibrated linear modules (127 demonstration reservoirs).
  - Irrigation    — calibrated seasonal constant releases (64 demonstration reservoirs).

All three share a conceptual S-curve that provides a 365-element typical-storage
array (S_ty) pre-computed at init.  At runtime the method compares current storage
and inflow against S_ty and percentile thresholds to select a release module.

References
----------
Zhao et al. (2025) C-GDROM reference code (github.com/fzfz12138/C-GDROM)
  cgdrom_general.py, cgdrom_fc.py, cgdrom_irr.py, conceptual_s_curve.py

Unit conventions
----------------
All values stored in CgdromParams are in SI units (m³, m³/s, s⁻¹).
Parameter files on disk use MCM for storage and m³/s for flows (see
preprocess/prepare_cgdrom_flow_stats.py for the conversion details).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from datetime import datetime

from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_ALPHA_DEFAULT = 2.0    # maximum upward ramp factor (daily); per-reservoir override via ALPHA column
_BETA_DEFAULT  = 0.5   # minimum downward ramp factor (daily); per-reservoir override via BETA column
_SECS_PER_DAY = 86400.0
_MCM_TO_M3    = 1.0e6   # million m³ → m³


# --------------------------------------------------------------------------- #
# S-curve pre-computation
# --------------------------------------------------------------------------- #

# Non-leap-year month lengths, 0-indexed (index 0 = January)
_MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
# Last DOY of month m: _MONTH_LAST_DOY[0]=0 (sentinel), [1]=31, ..., [12]=365
_MONTH_LAST_DOY = np.concatenate([[0], np.cumsum(_MONTH_DAYS)])
# Month number (1-12) for each DOY 1-365 (0-indexed array, index i → DOY i+1)
_DOY_MONTH = np.array([m for m, nd in enumerate(_MONTH_DAYS, 1) for _ in range(nd)])
_DOYS = np.arange(1, 366)


def _compute_sty(a1: int, a2: int, a3: int, a4: int,
                 s_low: float, s_high: float) -> np.ndarray:
    """Pre-compute the 365-element typical-storage array for a four-piece S-curve.

    Parameters mirror the A1-A4 columns in cgdrom_storage_curve.csv.
    a1 ∈ {0..7}, a2 ∈ {2..10}, a3 ∈ {4..12}, a4 ∈ {0..12}.
    a1 == 0 means the low-season plateau extends to before January.

    Returns
    -------
    sty : np.ndarray, shape (365,), 0-indexed (index 0 = Jan 1), m³
    """
    month = _DOY_MONTH
    doy   = _DOYS

    # End-DOY of transition anchor months (using cumulative table)
    month1_end = int(_MONTH_LAST_DOY[a1])   # 0 when a1 == 0
    month3_end = int(_MONTH_LAST_DOY[a3])

    # Transition lengths: days strictly between the anchor months
    # (mirrors range(a1+1, a2) and range(a3+1, a4) in the reference code)
    trans1 = int(_MONTH_DAYS[a1:a2 - 1].sum()) if a2 > a1 + 1 else 0

    if a4 > a1:  # no year-wrap on the falling limb
        trans2 = int(_MONTH_DAYS[a3:a4 - 1].sum()) if a4 > a3 + 1 else 0
    else:         # year-wrap: falling limb crosses December → January
        a4_prev = a4 - 1 if a4 > 1 else 0
        trans2 = int(np.r_[_MONTH_DAYS[a3:12], _MONTH_DAYS[0:a4_prev]].sum())

    sty = np.full(365, s_low, dtype=np.float64)

    if a4 > a1:
        # high plateau
        sty[(month >= a2) & (month <= a3)] = s_high
        # rising limb
        if trans1 > 0:
            m = (month > a1) & (month < a2)
            sty[m] = s_low + (s_high - s_low) * (doy[m] - month1_end) / trans1
        # falling limb
        if trans2 > 0:
            m = (month > a3) & (month < a4)
            sty[m] = s_high + (s_low - s_high) * (doy[m] - month3_end) / trans2
    else:
        # low plateau: month ≤ a1 AND month ≥ a4  (wraps around year-end)
        # high plateau
        sty[(month >= a2) & (month <= a3)] = s_high
        # rising limb
        if trans1 > 0:
            m = (month > a1) & (month < a2)
            sty[m] = s_low + (s_high - s_low) * (doy[m] - month1_end) / trans1
        # falling limb — straddles year-end; DOY adjustment for the wrap
        if trans2 > 0:
            m = (month > a3) | (month < a4)
            doy_adj = np.where(
                doy[m] >= month3_end,
                doy[m] - month3_end,
                doy[m] + 365 - month3_end,
            )
            sty[m] = s_high + (s_low - s_high) * doy_adj / trans2

    return sty


# --------------------------------------------------------------------------- #
# Per-reservoir parameter container
# --------------------------------------------------------------------------- #

class CgdromParams:
    """All per-reservoir parameters for the C-GDROM daily release computation.

    Constructed once at model init by load_cgdrom_params() and stored in
    grid.cgdrom_params keyed by GRanD ID (int).  All values are in SI units
    (m³, m³/s, s⁻¹) except size_ratio, alpha, and beta (dimensionless).
    """

    __slots__ = (
        'cgdrom_type',
        'sty',
        # storage limits (m³)
        's_cap', 's_dead', 's_flood_cap', 'size_ratio',
        # inflow statistics (m³/s)
        'i99', 'i80', 'i50', 'i30', 'i10', 'i_mean',
        # General Model derived constants (m³/s)
        'q1', 'q2', 'q3', 'r_flood',
        # FC module parameters (SI units; NaN → use General Model fallback)
        'm1_r',                                   # m³/s — Module 1 constant
        'm2a_coef', 'm2a_intercept', 'm2a_uses_inflow',
        'm2b_coef', 'm2b_intercept',              # Module 2b (storage-diff feature)
        'm3_r',                                   # m³/s — Module 3 / IRR irrigation season
        # IRR module parameters
        'doy_d1', 'doy_d2', 'doy_d3',
        'm4_r', 'm5_r',                           # m³/s — IRR spring / winter constants
        # ramping constraint factors (dimensionless)
        'alpha', 'beta',
    )

    def __init__(self, cgdrom_type, sty,
                 s_cap, s_dead, s_flood_cap, size_ratio,
                 i99, i80, i50, i30, i10, i_mean,
                 q1, q2, q3, r_flood,
                 m1_r=np.nan, m2a_coef=np.nan, m2a_intercept=np.nan,
                 m2a_uses_inflow=False, m2b_coef=np.nan, m2b_intercept=np.nan,
                 m3_r=np.nan, doy_d1=0, doy_d2=0, doy_d3=0, m4_r=np.nan, m5_r=np.nan,
                 alpha=_ALPHA_DEFAULT, beta=_BETA_DEFAULT):
        self.cgdrom_type = cgdrom_type
        self.sty = sty
        self.s_cap = s_cap
        self.s_dead = s_dead
        self.s_flood_cap = s_flood_cap
        self.size_ratio = size_ratio
        self.i99 = i99
        self.i80 = i80
        self.i50 = i50
        self.i30 = i30
        self.i10 = i10
        self.i_mean = i_mean
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.r_flood = r_flood
        self.m1_r = m1_r
        self.m2a_coef = m2a_coef
        self.m2a_intercept = m2a_intercept
        self.m2a_uses_inflow = m2a_uses_inflow
        self.m2b_coef = m2b_coef
        self.m2b_intercept = m2b_intercept
        self.m3_r = m3_r
        self.doy_d1 = doy_d1
        self.doy_d2 = doy_d2
        self.doy_d3 = doy_d3
        self.m4_r = m4_r
        self.m5_r = m5_r
        self.alpha = alpha
        self.beta = beta


# --------------------------------------------------------------------------- #
# Module prediction functions (one per C-GDROM variant)
# --------------------------------------------------------------------------- #

def _predict_general_daily(p: CgdromParams, it: float, st0: float,
                           rt0: float, s_ty: float) -> float:
    """C-GDROM General Model daily release (m³/s).

    Direct port of predict_general_daily() from cgdrom_general.py.
    Storage adjustment `Rt - (S_ty - S)` is converted from the reference's
    implicit MCM/day basis to an explicit m³/s basis via division by
    _SECS_PER_DAY.  Water balance update (St) is omitted; regulation() owns
    storage propagation in mosartwmpy.
    """
    # Module 1: major flood (It ≥ I99)
    if it >= p.i99:
        rt = p.r_flood
        if st0 < s_ty:
            rt -= (s_ty - st0) / _SECS_PER_DAY
        else:
            denom = p.s_cap - s_ty
            r2f = ((st0 - s_ty) / denom * (p.q1 - p.q2) + p.q2) if denom > 0 else p.q2
            if rt > r2f:
                rt = r2f

    # Module 2: above storage target
    elif st0 >= s_ty:
        denom = p.s_cap - s_ty
        rt = ((st0 - s_ty) / denom * (p.q1 - p.q2) + p.q2) if denom > 0 else p.q2

    # Module 3: below target, above dead storage
    elif st0 >= p.s_dead:
        rt = p.q3

    # Below dead storage
    else:
        rt = min(it, p.i10)

    # Spill guard
    if st0 > p.s_flood_cap:
        rt = max(it, rt)

    # Ramping constraints
    if rt > p.alpha * rt0 and rt0 > 0.0 and it < p.i80:
        rt = p.alpha * rt0
    if rt < p.beta * rt0:
        rt = p.beta * rt0
    return max(rt, 0.0)


def _predict_fc_daily(p: CgdromParams, it: float, st0: float,
                      rt0: float, s_ty: float) -> float:
    """C-GDROM Flood Control Model daily release (m³/s).

    Calibrated module parameters in CgdromParams.  NaN coefficient →
    falls back to the General Model interpolation formula for that module.
    """
    # Module 1: major flood (It ≥ I99)
    if it >= p.i99:
        rt = p.r_flood if np.isnan(p.m1_r) else min(p.m1_r, it)
        if st0 < s_ty:
            rt -= (s_ty - st0) / _SECS_PER_DAY

    # Module 2a: above target, high flow (I80 ≤ It < I99)
    elif it >= p.i80 and st0 >= s_ty:
        if np.isnan(p.m2a_coef):
            denom = p.s_flood_cap - s_ty
            rt = ((st0 - s_ty) / denom * (p.q1 - p.q2) + p.q2) if denom > 0 else p.q2
        elif p.m2a_uses_inflow:
            rt = min(p.m2a_coef * it + p.m2a_intercept, p.i99)
        else:
            rt = min(p.m2a_coef * (st0 - s_ty) + p.m2a_intercept, p.i99)

    # Module 2b: above target, low flow (It < I80)
    elif st0 >= s_ty:
        if np.isnan(p.m2b_coef):
            denom = p.s_flood_cap - s_ty
            rt = ((st0 - s_ty) / denom * (p.q1 - p.q2) + p.q2) if denom > 0 else p.q2
        else:
            rt = min(p.m2b_coef * (st0 - s_ty) + p.m2b_intercept, p.i99)

    # Module 3: below target, above dead storage
    elif st0 >= p.s_dead:
        rt = p.q3 if np.isnan(p.m3_r) else p.m3_r

    # Below dead storage
    else:
        rt = min(it, p.i10)

    # Spill guard
    if st0 > p.s_flood_cap:
        rt = max(it, rt)

    # Ramping constraints
    if rt > p.alpha * rt0 and rt0 > 0.0 and it < p.i80:
        rt = p.alpha * rt0
    if rt < p.beta * rt0:
        rt = p.beta * rt0
    return max(rt, 0.0)


def _predict_irr_daily(p: CgdromParams, it: float, st0: float,
                       rt0: float, doy: int, s_ty: float) -> float:
    """C-GDROM Irrigation Model daily release (m³/s).

    Module transitions are keyed to DOY breakpoints (DOY_D1/D2/D3) and the
    seasonal storage target S_ty.  Module 2 (high-flow) uses the General
    Model formula and requires no calibration.
    """
    def _spill_boost(m_r: float) -> float:
        """Above-target boost: max(calibrated_R, general-model interpolation)."""
        denom = p.s_flood_cap - s_ty
        r_gen = ((st0 - s_ty) / denom * (p.i99 - p.q2) + p.q2) if denom > 0 else p.q2
        return max(m_r, r_gen)

    # Module 1: major flood
    if it >= p.i99:
        rt = p.r_flood if np.isnan(p.m1_r) else min(p.m1_r, it)
        if st0 < s_ty:
            rt -= (s_ty - st0) / _SECS_PER_DAY

    # Module 2: high flow (I80 ≤ It < I99) — general-model formula; no calibrated params
    elif it >= p.i80:
        denom = p.s_flood_cap - s_ty
        rt = max(
            ((st0 - s_ty) / denom * (p.i99 - p.q2) + p.q2) if denom > 0 else p.q2,
            p.q2,
        )
        if st0 < s_ty:
            rt -= (s_ty - st0) / _SECS_PER_DAY

    # Module 3: irrigation season  (DOY_d3 ≤ DOY < DOY_d2)
    elif doy >= p.doy_d3 and doy < p.doy_d2:
        rt = _spill_boost(p.m3_r) if st0 > s_ty else p.m3_r

    # Module 4: spring high-flow season  (DOY_d1 ≤ DOY < DOY_d3)
    elif doy >= p.doy_d1 and doy < p.doy_d3:
        rt = _spill_boost(p.m4_r) if st0 > s_ty else p.m4_r

    # Module 5: winter low-flow season (default)
    else:
        rt = _spill_boost(p.m5_r) if st0 > s_ty else p.m5_r

    # Below dead storage override
    if st0 < p.s_dead:
        rt = min(it, p.i10)

    # Spill guard
    if st0 > p.s_flood_cap:
        rt = max(it, rt)

    # Ramping constraints
    if rt > p.alpha * rt0 and rt0 > 0.0 and it < p.i80:
        rt = p.alpha * rt0
    if rt < p.beta * rt0:
        rt = p.beta * rt0
    return max(rt, 0.0)


# --------------------------------------------------------------------------- #
# Parameter loading (called once at model init from reservoirs/grid.py)
# --------------------------------------------------------------------------- #

def load_cgdrom_params(config, reservoir_ids: np.ndarray,
                       reservoir_storage_capacity: np.ndarray,
                       cgdrom_indices) -> dict:
    """Load C-GDROM parameters for all eligible reservoirs.

    Called from _resolve_release_methods() in reservoirs/grid.py after the
    resolution loop has identified which cells will use C-GDROM.

    Parameters
    ----------
    config : Benedict
    reservoir_ids : np.ndarray
        GRanD IDs aligned to active-cell indices (float, NaN for non-reservoir cells).
    reservoir_storage_capacity : np.ndarray
        Active-capacity in m³ (grid.reservoir_storage_capacity).
    cgdrom_indices : array-like of int
        Active-cell indices where C-GDROM has been resolved.

    Returns
    -------
    dict : {grand_id (int): CgdromParams}
    """
    cgdrom_cfg = config.get('water_management.reservoirs.cgdrom', {}) or {}

    # Required: flow statistics
    flow_path = cgdrom_cfg.get('flow_stats.path')
    if not flow_path:
        raise ValueError(
            "C-GDROM is enabled but water_management.reservoirs.cgdrom.flow_stats.path "
            "is not set."
        )
    flow_df = pd.read_parquet(flow_path).set_index('GRAND_ID')

    # Required: storage curve parameters
    curve_path = cgdrom_cfg.get('storage_curve.path')
    if not curve_path:
        raise ValueError(
            "C-GDROM is enabled but water_management.reservoirs.cgdrom.storage_curve.path "
            "is not set."
        )
    curve_df = pd.read_csv(curve_path).set_index('GRAND_ID')

    # Optional: FC and IRR module files
    fc_path  = cgdrom_cfg.get('fc_modules.path')
    irr_path = cgdrom_cfg.get('irr_modules.path')
    fc_df  = pd.read_csv(fc_path).set_index('GRAND_ID')   if fc_path  and Path(fc_path).exists()  else None
    irr_df = pd.read_csv(irr_path).set_index('GRAND_ID')  if irr_path and Path(irr_path).exists() else None

    params = {}
    loaded_gids = set()
    for i in cgdrom_indices:
        i = int(i)
        gid = int(reservoir_ids[i]) if np.isfinite(float(reservoir_ids[i])) else -1
        if gid <= 0 or gid not in flow_df.index or gid not in curve_df.index:
            continue
        if gid in loaded_gids:
            continue
        loaded_gids.add(gid)

        # ---- Inflow statistics (m³/s; already converted in preprocess script) ----
        f = flow_df.loc[gid]
        i99   = float(f['I99'])
        i80   = float(f['I80'])
        i50   = float(f['I50'])
        i30   = float(f['I30'])
        i10   = float(f['I10'])
        i_mean = float(f['I_MEAN'])

        # ---- Storage curve ----
        c = curve_df.loc[gid]
        s_cap = float(reservoir_storage_capacity[i])   # from the mosartwmpy grid (m³)
        s_dead_raw = float(c['S_DEAD']) if pd.notna(c['S_DEAD']) else np.nan
        s_dead = s_dead_raw * _MCM_TO_M3 if not np.isnan(s_dead_raw) else 0.0
        s_flood_raw = float(c['S_FLOOD_CAP']) if pd.notna(c['S_FLOOD_CAP']) else np.nan
        s_flood_cap = (s_flood_raw * _MCM_TO_M3) if not np.isnan(s_flood_raw) else 0.99 * s_cap

        s_low    = float(c['S_A4_A1']) * _MCM_TO_M3 if pd.notna(c['S_A4_A1']) else 0.0
        s_high   = float(c['S_A2_A3']) * _MCM_TO_M3 if pd.notna(c['S_A2_A3']) else s_cap
        s_median = float(c['S_MEDIAN']) * _MCM_TO_M3 if pd.notna(c['S_MEDIAN']) else (s_low + s_high) / 2.0

        # S-curve: 365-element array (m³)
        curve_shape = str(c['CURVE_SHAPE']).lower()
        if curve_shape == 'single' or pd.isna(c.get('A1', np.nan)):
            sty = np.full(365, s_median, dtype=np.float64)
        else:
            a1, a2, a3, a4 = int(c['A1']), int(c['A2']), int(c['A3']), int(c['A4'])
            sty = _compute_sty(a1, a2, a3, a4, s_low, s_high)

        # ---- Size ratio (computed from the grid's S_cap, not the CSV S_cap) ----
        s_min = min(s_low, s_high)
        i_mean_m3 = i_mean * _SECS_PER_DAY  # m³/s × s/day → m³/day
        size_ratio = (s_cap - s_min) / (i_mean_m3 * 365) if i_mean > 0 else 0.0

        # ---- General Model derived constants ----
        q1 = i99
        if size_ratio > 0.4:
            q2, q3 = i_mean, i50
        else:
            q2, q3 = i50, i30
        r_flood = max((1.0 - 1.75 * size_ratio) * i99, i_mean)

        # ---- Determine C-GDROM tier: FC > IRR > General ----
        cgdrom_type = 'general'
        extra = {}

        if fc_df is not None and gid in fc_df.index:
            cgdrom_type = 'flood_control'
            r = fc_df.loc[gid]
            extra = dict(
                m1_r            = _safe_float(r['M1_R']),
                m2a_coef        = _safe_float(r['M2A_COEF']),
                m2a_intercept   = _safe_float(r['M2A_INTERCEPT']),
                m2a_uses_inflow = bool(r['M2A_USES_INFLOW']),
                m2b_coef        = _safe_float(r['M2B_COEF']),
                m2b_intercept   = _safe_float(r['M2B_INTERCEPT']),
                m3_r            = _safe_float(r['M3_R']),
            )
            tier_row = r
        elif irr_df is not None and gid in irr_df.index:
            cgdrom_type = 'irrigation'
            r = irr_df.loc[gid]
            extra = dict(
                m1_r   = _safe_float(r['M1_R']),
                m3_r   = _safe_float(r['M3_R']),
                doy_d1 = int(r['DOY_D1']),
                doy_d2 = int(r['DOY_D2']),
                doy_d3 = int(r['DOY_D3']),
                m4_r   = _safe_float(r['M4_R']),
                m5_r   = _safe_float(r['M5_R']),
            )
            tier_row = r
        else:
            tier_row = c

        # Ramping factors: ALPHA/BETA column in tier file overrides module default.
        # Missing column or NaN value → fall back to _ALPHA_DEFAULT / _BETA_DEFAULT.
        alpha_raw = _safe_float(tier_row.get('ALPHA', np.nan))
        beta_raw  = _safe_float(tier_row.get('BETA',  np.nan))
        extra['alpha'] = alpha_raw if not np.isnan(alpha_raw) else _ALPHA_DEFAULT
        extra['beta']  = beta_raw  if not np.isnan(beta_raw)  else _BETA_DEFAULT

        params[gid] = CgdromParams(
            cgdrom_type=cgdrom_type, sty=sty,
            s_cap=s_cap, s_dead=s_dead, s_flood_cap=s_flood_cap, size_ratio=size_ratio,
            i99=i99, i80=i80, i50=i50, i30=i30, i10=i10, i_mean=i_mean,
            q1=q1, q2=q2, q3=q3, r_flood=r_flood,
            **extra,
        )
        logging.debug(
            "C-GDROM: loaded %s params for GRAND_ID=%d",
            cgdrom_type, gid,
        )

    n_loaded = len(params)
    n_requested = len(list(cgdrom_indices))
    if n_loaded < n_requested:
        logging.warning(
            "C-GDROM: loaded params for %d of %d eligible reservoirs; "
            "%d had no valid GRAND_ID or were missing curve rows "
            "(eligibility should have been filtered at init — check _init_cgdrom_data).",
            n_loaded, n_requested, n_requested - n_loaded,
        )
    else:
        logging.info("C-GDROM: loaded params for %d reservoirs.", n_loaded)

    return params


def _safe_float(val) -> float:
    """Return float, or NaN for any non-finite or non-numeric value."""
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------- #
# Daily release function (called once per day from reservoirs/release.py)
# --------------------------------------------------------------------------- #

def cgdrom_release(state: State, grid: Grid, current_time: datetime) -> None:
    """Update state.reservoir_release [m³/s] for C-GDROM-eligible reservoirs.

    Fires once per day at midnight, after gdrom_release() has already run.
    Writes only to cells where grid.uses_cgdrom is True, giving C-GDROM final
    say over any earlier daily update.

    Previous-day release is tracked in state.reservoir_cgdrom_prev_release for the
    per-reservoir ramping constraints (p.alpha upward, p.beta downward; defaults 2 / 0.5).
    That array is zero-initialised at model start, so the first call has rt0 = 0 and ramp
    constraints are skipped.  Because it lives in State it is saved and restored with
    restart files, so a restarted run produces the same ramp behaviour as a continuous one.
    """
    # Clamp leap-year DOY 366 to 365; CGDROM params are on a 365-day calendar.
    doy = min(current_time.timetuple().tm_yday, 365)

    indices = np.where(grid.uses_cgdrom)[0]
    if indices.size == 0:
        return

    for i in indices:
        grand_id = int(grid.reservoir_id[i])
        p = grid.cgdrom_params.get(grand_id)
        if p is None:
            # params not loaded for this reservoir; skip silently
            continue

        it  = float(state.channel_inflow_upstream[i])
        it  = max(it, 0.0)
        st0 = float(state.reservoir_storage[i])
        rt0 = float(state.reservoir_cgdrom_prev_release[i])

        # DOY-indexed typical storage (0-indexed array, DOY 1 → index 0)
        s_ty = float(p.sty[doy - 1])

        if p.cgdrom_type == 'flood_control':
            rt = _predict_fc_daily(p, it, st0, rt0, s_ty)
        elif p.cgdrom_type == 'irrigation':
            rt = _predict_irr_daily(p, it, st0, rt0, doy, s_ty)
        else:
            rt = _predict_general_daily(p, it, st0, rt0, s_ty)

        state.reservoir_release[i] = rt
        state.reservoir_cgdrom_prev_release[i] = rt
