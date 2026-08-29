"""GDROM (Generic Data-Driven Reservoir Operation Model) release method.

Implements the two-stage inference pipeline (CART module-condition classifier →
Decision Tree release regressor) described in:

  Li et al. (2024) "Uncovering Historical Reservoir Operation Rules..." WRR.
  GDROM v2.0.0 dataset documentation (s41597-025-06162-7).

Rule files are pre-parsed at model init by load_gdrom_rules() in reservoirs/grid.py
and stored on grid.gdrom_rules as a dict keyed by GRanD ID. All Inflow/Storage
thresholds and Release leaf values are converted from acre-feet units to m³ (m³/s)
at parse time, so inference operates entirely in SI units matching mosartwmpy state.

PDSI is a required input for the module-condition classifier. It is loaded and
validated at init via load_pdsi() in reservoirs/grid.py; at runtime the lookup
grid.pdsi_lookup(state_name, year, month) returns the relevant value.
"""

import logging
import re

import numpy as np

from datetime import datetime

from mosartwmpy.grid.grid import Grid
from mosartwmpy.state.state import State


# --------------------------------------------------------------------------- #
# Unit conversion constants
# --------------------------------------------------------------------------- #

# 1 acre-foot = 1233.48 m³
_ACFT_TO_M3 = 1233.48
# 1 acre-foot/day = 1233.48 / 86400 m³/s
_ACFT_DAY_TO_M3S = 1233.48 / 86400.0

# --------------------------------------------------------------------------- #
# Variable index mapping used inside parsed rule tuples
# --------------------------------------------------------------------------- #
# CT rules use all four variables; module rules only use Inflow (0) and Storage (1)
_VAR_IDX = {"Inflow": 0, "Storage": 1, "DOY": 2, "PDSI": 3}

# Regex for a single condition term: (Variable OP threshold)
_COND_RE = re.compile(r"\((\w+) (<=|>) ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)")


# --------------------------------------------------------------------------- #
# Rule file parsers
# --------------------------------------------------------------------------- #

def _parse_rule_lines(path, is_ct: bool, convert_release: bool = False):
    """Parse one GDROM rule file into a list of (conditions, value) tuples.

    Each tuple represents one root-to-leaf path in the compiled decision tree.
    The first path whose conditions all evaluate to True is the match.

    Parameters
    ----------
    path : str or Path
        Rule file path.
    is_ct : bool
        True for module-condition (CT) files (terminal keyword 'module');
        False for release-module files (terminal keyword 'Release').
    convert_release : bool
        When True, multiply Release leaf values by _ACFT_DAY_TO_M3S.
        Has no effect for CT files.

    Returns
    -------
    list of (tuple_of_conditions, float)
        Each conditions element is a tuple of (var_idx: int, op_le: bool, threshold: float).
        For CT files the float is the integer module id (stored as float for uniformity).
    """
    parsed = []
    terminal_kw = "module" if is_ct else "Release"
    terminal_pat = re.compile(rf"then {terminal_kw}: ([+-]?\d+(?:\.\d+)?)")

    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            # parse terminal value
            t_match = terminal_pat.search(line)
            if t_match is None:
                continue
            value = float(t_match.group(1))
            if convert_release:
                value *= _ACFT_DAY_TO_M3S

            # parse all conditions
            conditions = []
            for m in _COND_RE.finditer(line):
                var_name, op, threshold_str = m.group(1), m.group(2), m.group(3)
                var_idx = _VAR_IDX.get(var_name)
                if var_idx is None:
                    # unknown variable — skip line rather than silently misroute
                    conditions = None
                    break
                threshold = float(threshold_str)
                # convert Inflow and Storage thresholds; DOY and PDSI are dimensionless
                if var_name == "Inflow":
                    threshold *= _ACFT_DAY_TO_M3S
                elif var_name == "Storage":
                    threshold *= _ACFT_TO_M3
                op_le = (op == "<=")
                conditions.append((var_idx, op_le, threshold))

            if conditions is not None:
                parsed.append((tuple(conditions), value))

    return parsed


def load_gdrom_rules(rules_dir, grand_ids):
    """Parse all rule files for the given GRanD IDs.

    Called once at model init from reservoirs/grid.py. Reads files under
    rules_dir/module_conditions/ and rules_dir/modules/.

    Parameters
    ----------
    rules_dir : str or Path
        Directory containing module_conditions/ and modules/ subdirectories.
    grand_ids : iterable of int
        GRanD IDs for which to load rules (i.e. the GDROM-eligible reservoir IDs).

    Returns
    -------
    dict : {grand_id (int): {'ct': list or None, 'modules': dict[int, list]}}
        'ct' is None for single-module reservoirs (no CT file).
    """
    from pathlib import Path

    rules_dir = Path(rules_dir)
    ct_dir = rules_dir / "module_conditions"
    mod_dir = rules_dir / "modules"

    parsed_rules = {}
    for gid in grand_ids:
        gid = int(gid)

        # load module files (at least one must exist)
        modules = {}
        m_idx = 0
        while True:
            mod_path = mod_dir / f"{gid}_{m_idx}.txt"
            if not mod_path.exists():
                break
            modules[m_idx] = _parse_rule_lines(mod_path, is_ct=False, convert_release=True)
            m_idx += 1

        if not modules:
            # no module files at all — skip this reservoir
            continue

        # load CT file (absent for single-module reservoirs)
        ct_path = ct_dir / f"{gid}.txt"
        ct_rules = _parse_rule_lines(ct_path, is_ct=True) if ct_path.exists() else None

        parsed_rules[gid] = {"ct": ct_rules, "modules": modules}

    return parsed_rules


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #

def _evaluate_rules(rule_lines, vals):
    """Walk rule_lines in order; return value of first matching path.

    Parameters
    ----------
    rule_lines : list of (conditions_tuple, float)
    vals : tuple of float
        Input variable values indexed by _VAR_IDX.

    Returns
    -------
    float or None — None if no path matched (should not happen for valid trees).
    """
    for conditions, value in rule_lines:
        match = True
        for var_idx, op_le, threshold in conditions:
            v = vals[var_idx]
            if op_le:
                if not (v <= threshold):
                    match = False
                    break
            else:
                if not (v > threshold):
                    match = False
                    break
        if match:
            return value
    return None


# --------------------------------------------------------------------------- #
# Daily release function
# --------------------------------------------------------------------------- #

def gdrom_release(state: State, grid: Grid, current_time: datetime) -> None:
    """Update state.reservoir_release [m³/s] for GDROM-eligible reservoirs.

    Fires once per day at midnight, after istarf_release() has already run
    for ISTARF reservoirs. Writes only to cells where grid.uses_gdrom is True.

    Fallback behaviour when inference fails for a reservoir (no CT branch match,
    missing PDSI, etc.): retain the existing state.reservoir_release value and
    log a warning (once per day per reservoir to avoid flooding the log).
    """
    doy = current_time.timetuple().tm_yday
    year = current_time.year
    month = current_time.month

    indices = np.where(grid.uses_gdrom)[0]
    if indices.size == 0:
        return

    for i in indices:
        grand_id = int(grid.reservoir_id[i])
        rules = grid.gdrom_rules.get(grand_id)
        if rules is None:
            # should not happen after init validation, but guard defensively
            continue

        inflow_m3s = float(state.channel_inflow_upstream[i])
        storage_m3 = float(state.reservoir_storage[i])

        # guard against negative inflow (net inflow can be negative in GDROM
        # training data, but channel_inflow_upstream should always be ≥ 0)
        if inflow_m3s < 0.0:
            inflow_m3s = 0.0

        # PDSI lookup
        state_name = str(grid.reservoir_state_name[i])
        try:
            pdsi = grid.pdsi_lookup(state_name, year, month)
        except KeyError:
            logging.warning(
                "GDROM: no PDSI value for state '%s' year=%d month=%d "
                "(reservoir GRAND_ID=%d); retaining existing release target",
                state_name, year, month, grand_id,
            )
            continue

        vals = (inflow_m3s, storage_m3, float(doy), float(pdsi))

        # stage 1: module-condition classifier
        ct_rules = rules["ct"]
        if ct_rules is None:
            # single-module reservoir
            module_id = 0
        else:
            result = _evaluate_rules(ct_rules, vals)
            if result is None:
                logging.warning(
                    "GDROM: CT classifier found no matching branch for "
                    "GRAND_ID=%d on %s (inflow=%.3f m³/s, storage=%.1f m³, "
                    "DOY=%d, PDSI=%.2f); retaining existing release target",
                    grand_id, current_time.date(), inflow_m3s, storage_m3, doy, pdsi,
                )
                continue
            module_id = int(result)

        # stage 2: release regressor
        module_rules = rules["modules"].get(module_id)
        if module_rules is None:
            logging.warning(
                "GDROM: module %d not found for GRAND_ID=%d on %s; "
                "retaining existing release target",
                module_id, grand_id, current_time.date(),
            )
            continue

        release = _evaluate_rules(module_rules, vals)
        if release is None:
            logging.warning(
                "GDROM: release module %d found no matching branch for "
                "GRAND_ID=%d on %s; retaining existing release target",
                module_id, grand_id, current_time.date(),
            )
            continue

        # release is already in m³/s (converted at parse time)
        state.reservoir_release[i] = release
