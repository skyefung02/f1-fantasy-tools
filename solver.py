"""
F1 Fantasy Integer Linear Programming Solver
=============================================
Maximises expected points subject to:
  - Exactly 5 drivers selected
  - Exactly 2 constructors selected
  - Exactly 1 turbo driver (scores 2x points)
  - Total price <= budget (default 100)

CSV format expected:
  name, type (driver|constructor), price, expected_points
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import pulp

SETTINGS_FILE = Path(__file__).parent / "settings.json"
DEFAULT_SETTINGS = {
    "budget": 100.0,
    "data_file": "sample_data.csv",
    "locked": [],
    "banned": [],
    "top_n": 5,
    "portfolio_n": None,
    "portfolio_max_overlap": 5,
    "pts_per_1m_per_race": 0.0,
    "remaining_races": 23,
}


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS


# ── ILP solver ────────────────────────────────────────────────────────────────

def solve(
    df: pd.DataFrame,
    budget: float = 100.0,
    locked: list[str] | None = None,
    banned: list[str] | None = None,
    exclusions: list[dict] | None = None,
    overlap_constraints: list[tuple[dict, int]] | None = None,
    budget_pts_weight: float = 0.0,
) -> dict:
    """
    Run the F1 Fantasy ILP optimiser.

    Parameters
    ----------
    df                  : DataFrame with columns [name, type, price, expected_points]
                          and optionally [xDeltaPrice]
    budget              : Total spend cap (default 100)
    locked              : List of codes that MUST be in the team
    banned              : List of codes that MUST NOT be in the team
    exclusions          : List of previous result dicts whose exact team+turbo combo must not repeat
    overlap_constraints : List of (prev_result, max_overlap) pairs. Overlap is counted over
                          8 decisions (5 driver selections + 2 constructor selections +
                          1 turbo designation); max_overlap caps how many can match.
    budget_pts_weight   : Points value of a 1M price increase over all remaining races
                          (= pts_per_1m_per_race × remaining_races). 0 disables budget optimisation.
    """
    locked = {c.upper() for c in (locked or [])}
    banned = {c.upper() for c in (banned or [])}

    overlap = locked & banned
    if overlap:
        raise ValueError(f"Codes appear in both locked and banned: {overlap}")

    drivers = df[df["type"] == "driver"].reset_index(drop=True)
    constructors = df[df["type"] == "constructor"].reset_index(drop=True)

    all_codes = set(df["name"].str.upper())
    unknown = (locked | banned) - all_codes
    if unknown:
        raise ValueError(f"Codes not found in data: {unknown}")

    n_d = len(drivers)
    n_c = len(constructors)

    locked_drivers = locked & set(drivers["name"].str.upper())
    locked_constructors = locked & set(constructors["name"].str.upper())

    if len(locked_drivers) > 5:
        raise ValueError(f"Cannot lock more than 5 drivers (got {len(locked_drivers)}).")
    if len(locked_constructors) > 2:
        raise ValueError(f"Cannot lock more than 2 constructors (got {len(locked_constructors)}).")

    if n_d < 5:
        raise ValueError(f"Need at least 5 drivers in the data, got {n_d}.")
    if n_c < 2:
        raise ValueError(f"Need at least 2 constructors in the data, got {n_c}.")

    prob = pulp.LpProblem("F1_Fantasy_Optimizer", pulp.LpMaximize)

    # ── Decision variables ────────────────────────────────────────────────────
    # x_d[i] = 1 if driver i is in the team
    x_d = [pulp.LpVariable(f"driver_{i}", cat="Binary") for i in range(n_d)]
    # x_c[j] = 1 if constructor j is in the team
    x_c = [pulp.LpVariable(f"constructor_{j}", cat="Binary") for j in range(n_c)]
    # t_d[i] = 1 if driver i is the turbo driver (only one allowed)
    t_d = [pulp.LpVariable(f"turbo_{i}", cat="Binary") for i in range(n_d)]

    # ── Objective ─────────────────────────────────────────────────────────────
    # Turbo driver earns 2x xPts → regular pts + bonus pts (turbo does NOT double price value)
    # Combined score = xPts + xDeltaPrice * budget_pts_weight (if enabled)
    has_delta = "xDeltaPrice" in df.columns and budget_pts_weight != 0.0

    driver_pts = pulp.lpSum(
        (x_d[i] + t_d[i]) * drivers.loc[i, "expected_points"] for i in range(n_d)
    )
    constructor_pts = pulp.lpSum(
        x_c[j] * constructors.loc[j, "expected_points"] for j in range(n_c)
    )
    if has_delta:
        driver_pts += pulp.lpSum(
            x_d[i] * drivers.loc[i, "xDeltaPrice"] * budget_pts_weight for i in range(n_d)
        )
        constructor_pts += pulp.lpSum(
            x_c[j] * constructors.loc[j, "xDeltaPrice"] * budget_pts_weight for j in range(n_c)
        )
    prob += driver_pts + constructor_pts

    # ── Constraints ───────────────────────────────────────────────────────────
    prob += pulp.lpSum(x_d) == 5, "exactly_5_drivers"
    prob += pulp.lpSum(x_c) == 2, "exactly_2_constructors"
    prob += pulp.lpSum(t_d) == 1, "exactly_1_turbo_driver"

    # Turbo pick must be one of the selected drivers
    for i in range(n_d):
        prob += t_d[i] <= x_d[i], f"turbo_must_be_selected_{i}"

    # Lock / ban constraints
    for i in range(n_d):
        code = drivers.loc[i, "name"].upper()
        if code in locked:
            prob += x_d[i] == 1, f"lock_driver_{code}"
        elif code in banned:
            prob += x_d[i] == 0, f"ban_driver_{code}"

    for j in range(n_c):
        code = constructors.loc[j, "name"].upper()
        if code in locked:
            prob += x_c[j] == 1, f"lock_constructor_{code}"
        elif code in banned:
            prob += x_c[j] == 0, f"ban_constructor_{code}"

    # Budget cap
    total_spend = pulp.lpSum(
        x_d[i] * drivers.loc[i, "price"] for i in range(n_d)
    ) + pulp.lpSum(
        x_c[j] * constructors.loc[j, "price"] for j in range(n_c)
    )
    prob += total_spend <= budget, "budget_cap"

    # ── Exclude previously found solutions ───────────────────────────────────
    # For each prior solution, force at least one pick (driver, constructor, or
    # turbo choice) to differ. The solution is identified by 8 binary vars
    # (5 x_d + 2 x_c + 1 t_d); constraining their sum <= 7 rules it out.
    for k, prev in enumerate(exclusions or []):
        prev_drivers = {d["name"].upper() for d in prev["drivers"]}
        prev_constructors = {c["name"].upper() for c in prev["constructors"]}
        prev_turbo = prev["turbo_driver"].upper()

        excl_vars = []
        for i in range(n_d):
            code = drivers.loc[i, "name"].upper()
            if code in prev_drivers:
                excl_vars.append(x_d[i])
            if code == prev_turbo:
                excl_vars.append(t_d[i])
        for j in range(n_c):
            if constructors.loc[j, "name"].upper() in prev_constructors:
                excl_vars.append(x_c[j])

        prob += pulp.lpSum(excl_vars) <= len(excl_vars) - 1, f"exclude_solution_{k}"

    # ── Pairwise overlap constraints ──────────────────────────────────────────
    # Each (prev_team, max_overlap) pair caps how many of the 8 decisions
    # (5 x_d + 2 x_c + 1 t_d) can match the previous team.
    for k, (prev, max_overlap) in enumerate(overlap_constraints or []):
        prev_drivers = {d["name"].upper() for d in prev["drivers"]}
        prev_constructors = {c["name"].upper() for c in prev["constructors"]}
        prev_turbo = prev["turbo_driver"].upper()

        overlap_vars = []
        for i in range(n_d):
            code = drivers.loc[i, "name"].upper()
            if code in prev_drivers:
                overlap_vars.append(x_d[i])
            if code == prev_turbo:
                overlap_vars.append(t_d[i])
        for j in range(n_c):
            if constructors.loc[j, "name"].upper() in prev_constructors:
                overlap_vars.append(x_c[j])

        prob += pulp.lpSum(overlap_vars) <= max_overlap, f"overlap_constraint_{k}"

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    if prob.status != 1:  # 1 = Optimal
        raise RuntimeError(f"Solver did not find an optimal solution. Status: {status}")

    # ── Parse results ─────────────────────────────────────────────────────────
    selected_drivers = []
    turbo_driver = None

    has_delta_col = "xDeltaPrice" in drivers.columns

    for i in range(n_d):
        if pulp.value(x_d[i]) > 0.5:
            row = drivers.loc[i]
            is_turbo = pulp.value(t_d[i]) > 0.5
            selected_drivers.append({
                "name": row["name"],
                "price": row["price"],
                "expected_points": row["expected_points"],
                "xDeltaPrice": float(row["xDeltaPrice"]) if has_delta_col else 0.0,
                "is_turbo": is_turbo,
            })
            if is_turbo:
                turbo_driver = row["name"]

    selected_constructors = []
    has_delta_col_c = "xDeltaPrice" in constructors.columns

    for j in range(n_c):
        if pulp.value(x_c[j]) > 0.5:
            row = constructors.loc[j]
            selected_constructors.append({
                "name": row["name"],
                "price": row["price"],
                "expected_points": row["expected_points"],
                "xDeltaPrice": float(row["xDeltaPrice"]) if has_delta_col_c else 0.0,
            })

    # Reorder: non-turbo first, turbo highlighted at end
    selected_drivers.sort(key=lambda d: d["is_turbo"])

    spent = sum(d["price"] for d in selected_drivers) + sum(
        c["price"] for c in selected_constructors
    )
    pts = sum(
        (d["expected_points"] * 2 if d["is_turbo"] else d["expected_points"])
        for d in selected_drivers
    ) + sum(c["expected_points"] for c in selected_constructors)

    budget_value = (
        sum(d["xDeltaPrice"] for d in selected_drivers)
        + sum(c["xDeltaPrice"] for c in selected_constructors)
    ) * budget_pts_weight

    return {
        "drivers": selected_drivers,
        "turbo_driver": turbo_driver,
        "constructors": selected_constructors,
        "total_price": round(spent, 1),
        "remaining_budget": round(budget - spent, 1),
        "total_points": round(pts, 2),
        "budget_value": round(budget_value, 2),
    }


# ── Top-N solver ──────────────────────────────────────────────────────────────

def solve_top_n(
    df: pd.DataFrame,
    budget: float = 100.0,
    locked: list[str] | None = None,
    banned: list[str] | None = None,
    n: int = 5,
    budget_pts_weight: float = 0.0,
) -> list[dict]:
    """Return the top-n distinct optimal teams, ranked by total points."""
    results = []
    for _ in range(n):
        try:
            result = solve(
                df, budget=budget, locked=locked, banned=banned,
                exclusions=results, budget_pts_weight=budget_pts_weight,
            )
        except RuntimeError:
            break
        results.append(result)
    return results


# ── Portfolio solver ──────────────────────────────────────────────────────────

def compute_overlap(team1: dict, team2: dict) -> int:
    """
    Count the number of matching decisions between two teams (out of 8 total):
      5 driver selections + 2 constructor selections + 1 turbo designation.
    A shared driver who is also the turbo in both teams counts as 2.
    """
    drivers1 = {d["name"].upper() for d in team1["drivers"]}
    drivers2 = {d["name"].upper() for d in team2["drivers"]}
    constructors1 = {c["name"].upper() for c in team1["constructors"]}
    constructors2 = {c["name"].upper() for c in team2["constructors"]}
    shared_drivers = len(drivers1 & drivers2)
    shared_constructors = len(constructors1 & constructors2)
    same_turbo = int(team1["turbo_driver"].upper() == team2["turbo_driver"].upper())
    return shared_drivers + shared_constructors + same_turbo


def solve_portfolio(
    df: pd.DataFrame,
    budget: float = 100.0,
    locked: list[str] | None = None,
    banned: list[str] | None = None,
    n_teams: int = 3,
    max_pairwise_overlap: int = 5,
    budget_pts_weight: float = 0.0,
) -> list[dict]:
    """
    Solve for n_teams distinct teams that together form a diversified portfolio.

    Each team after the first is constrained so its overlap with every previously
    solved team is at most max_pairwise_overlap (out of 8 decisions).

    Overlap is defined as:
      - 1 point per shared driver selection  (5 possible)
      - 1 point per shared constructor       (2 possible)
      - 1 point if the turbo driver matches  (1 possible)
      Total max = 8 (identical teams)

    The first team is solved unconstrained to establish the xPts ceiling.
    Subsequent teams show the xPts cost of each diversity step.
    """
    teams = []
    for _ in range(n_teams):
        constraints = [(t, max_pairwise_overlap) for t in teams]
        try:
            team = solve(
                df,
                budget=budget,
                locked=locked,
                banned=banned,
                overlap_constraints=constraints,
                budget_pts_weight=budget_pts_weight,
            )
        except RuntimeError:
            break
        teams.append(team)
    return teams


# ── Transfer solver ───────────────────────────────────────────────────────────

def solve_with_transfers(
    df: pd.DataFrame,
    current_driver_tlas: set[str],
    current_constructor_tlas: set[str],
    budget: float,
    locked: list[str] | None = None,
    banned: list[str] | None = None,
    overlap_constraints: list[tuple[dict, int]] | None = None,
    max_free_transfers: int = 2,
    penalty_per_transfer: int = 10,
    budget_pts_weight: float = 0.0,
) -> dict:
    """
    Solve for optimal transfers from a current team.

    Transfer counting covers drivers and constructors only (7 picks total).
    Changing the turbo designation is always free.

    Parameters
    ----------
    current_driver_tlas      : TLAs of the 5 drivers currently on the team
    current_constructor_tlas : TLAs of the 2 constructors currently on the team
    budget                   : Total available budget (team value + remaining cash)
    max_free_transfers       : Number of free transfers (default 2)
    penalty_per_transfer     : Points penalty per transfer beyond the free allowance
    """
    locked = {c.upper() for c in (locked or [])}
    banned = {c.upper() for c in (banned or [])}
    current_driver_tlas = {t.upper() for t in current_driver_tlas}
    current_constructor_tlas = {t.upper() for t in current_constructor_tlas}

    overlap = locked & banned
    if overlap:
        raise ValueError(f"Codes appear in both locked and banned: {overlap}")

    drivers = df[df["type"] == "driver"].reset_index(drop=True)
    constructors = df[df["type"] == "constructor"].reset_index(drop=True)

    all_codes = set(df["name"].str.upper())
    unknown = (locked | banned) - all_codes
    if unknown:
        raise ValueError(f"Codes not found in data: {unknown}")

    n_d = len(drivers)
    n_c = len(constructors)

    if n_d < 5:
        raise ValueError(f"Need at least 5 drivers, got {n_d}.")
    if n_c < 2:
        raise ValueError(f"Need at least 2 constructors, got {n_c}.")

    prob = pulp.LpProblem("F1_Fantasy_Transfers", pulp.LpMaximize)

    x_d = [pulp.LpVariable(f"driver_{i}", cat="Binary") for i in range(n_d)]
    x_c = [pulp.LpVariable(f"constructor_{j}", cat="Binary") for j in range(n_c)]
    t_d = [pulp.LpVariable(f"turbo_{i}", cat="Binary") for i in range(n_d)]
    # e = excess transfers beyond the free allowance (what gets penalised)
    e = pulp.LpVariable("excess_transfers", lowBound=0, cat="Integer")

    # ── Objective: gross points + budget value − penalty ──────────────────────
    has_delta = "xDeltaPrice" in df.columns and budget_pts_weight != 0.0

    driver_pts = pulp.lpSum(
        (x_d[i] + t_d[i]) * drivers.loc[i, "expected_points"] for i in range(n_d)
    )
    constructor_pts = pulp.lpSum(
        x_c[j] * constructors.loc[j, "expected_points"] for j in range(n_c)
    )
    if has_delta:
        driver_pts += pulp.lpSum(
            x_d[i] * drivers.loc[i, "xDeltaPrice"] * budget_pts_weight for i in range(n_d)
        )
        constructor_pts += pulp.lpSum(
            x_c[j] * constructors.loc[j, "xDeltaPrice"] * budget_pts_weight for j in range(n_c)
        )
    prob += driver_pts + constructor_pts - penalty_per_transfer * e

    # ── Standard constraints ──────────────────────────────────────────────────
    prob += pulp.lpSum(x_d) == 5, "exactly_5_drivers"
    prob += pulp.lpSum(x_c) == 2, "exactly_2_constructors"
    prob += pulp.lpSum(t_d) == 1, "exactly_1_turbo_driver"

    for i in range(n_d):
        prob += t_d[i] <= x_d[i], f"turbo_must_be_selected_{i}"

    for i in range(n_d):
        code = drivers.loc[i, "name"].upper()
        if code in locked:
            prob += x_d[i] == 1, f"lock_driver_{code}"
        elif code in banned:
            prob += x_d[i] == 0, f"ban_driver_{code}"

    for j in range(n_c):
        code = constructors.loc[j, "name"].upper()
        if code in locked:
            prob += x_c[j] == 1, f"lock_constructor_{code}"
        elif code in banned:
            prob += x_c[j] == 0, f"ban_constructor_{code}"

    prob += pulp.lpSum(
        x_d[i] * drivers.loc[i, "price"] for i in range(n_d)
    ) + pulp.lpSum(
        x_c[j] * constructors.loc[j, "price"] for j in range(n_c)
    ) <= budget, "budget_cap"

    # ── Transfer penalty ──────────────────────────────────────────────────────
    # kept  = number of current picks retained in the new team (out of 7)
    # n_transfers = 7 - kept
    # excess = max(0, n_transfers - max_free) = max(0, 7 - kept - max_free)
    # Linearised: e >= 7 - kept - max_free_transfers,  e >= 0 (already via lowBound)
    kept = pulp.lpSum(
        x_d[i] for i in range(n_d) if drivers.loc[i, "name"].upper() in current_driver_tlas
    ) + pulp.lpSum(
        x_c[j] for j in range(n_c) if constructors.loc[j, "name"].upper() in current_constructor_tlas
    )
    prob += e >= 7 - kept - max_free_transfers, "excess_transfers_lb"

    # ── Pairwise overlap constraints ──────────────────────────────────────────
    for k, (prev, max_overlap) in enumerate(overlap_constraints or []):
        prev_drivers = {d["name"].upper() for d in prev["drivers"]}
        prev_constructors = {c["name"].upper() for c in prev["constructors"]}
        prev_turbo = prev["turbo_driver"].upper()

        overlap_vars = []
        for i in range(n_d):
            code = drivers.loc[i, "name"].upper()
            if code in prev_drivers:
                overlap_vars.append(x_d[i])
            if code == prev_turbo:
                overlap_vars.append(t_d[i])
        for j in range(n_c):
            if constructors.loc[j, "name"].upper() in prev_constructors:
                overlap_vars.append(x_c[j])

        prob += pulp.lpSum(overlap_vars) <= max_overlap, f"overlap_constraint_{k}"

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        raise RuntimeError(
            f"Solver did not find an optimal solution. Status: {pulp.LpStatus[prob.status]}"
        )

    # ── Parse results ─────────────────────────────────────────────────────────
    selected_drivers = []
    turbo_driver = None
    new_driver_tlas = set()
    has_delta_col = "xDeltaPrice" in drivers.columns

    for i in range(n_d):
        if pulp.value(x_d[i]) > 0.5:
            row = drivers.loc[i]
            is_turbo = pulp.value(t_d[i]) > 0.5
            tla = row["name"].upper()
            selected_drivers.append({
                "name": row["name"],
                "price": row["price"],
                "expected_points": row["expected_points"],
                "xDeltaPrice": float(row["xDeltaPrice"]) if has_delta_col else 0.0,
                "is_turbo": is_turbo,
            })
            new_driver_tlas.add(tla)
            if is_turbo:
                turbo_driver = row["name"]

    selected_constructors = []
    new_constructor_tlas = set()
    has_delta_col_c = "xDeltaPrice" in constructors.columns

    for j in range(n_c):
        if pulp.value(x_c[j]) > 0.5:
            row = constructors.loc[j]
            selected_constructors.append({
                "name": row["name"],
                "price": row["price"],
                "expected_points": row["expected_points"],
                "xDeltaPrice": float(row["xDeltaPrice"]) if has_delta_col_c else 0.0,
            })
            new_constructor_tlas.add(row["name"].upper())

    selected_drivers.sort(key=lambda d: d["is_turbo"])

    spent = sum(d["price"] for d in selected_drivers) + sum(
        c["price"] for c in selected_constructors
    )
    gross_pts = sum(
        (d["expected_points"] * 2 if d["is_turbo"] else d["expected_points"])
        for d in selected_drivers
    ) + sum(c["expected_points"] for c in selected_constructors)

    budget_value = (
        sum(d["xDeltaPrice"] for d in selected_drivers)
        + sum(c["xDeltaPrice"] for c in selected_constructors)
    ) * budget_pts_weight

    n_transfers = max(0, round(pulp.value(e))) + max_free_transfers
    # Recompute n_transfers accurately from set differences (more reliable than ILP var)
    driver_out = sorted(current_driver_tlas - new_driver_tlas)
    driver_in  = sorted(new_driver_tlas - current_driver_tlas)
    constr_out = sorted(current_constructor_tlas - new_constructor_tlas)
    constr_in  = sorted(new_constructor_tlas - current_constructor_tlas)
    n_transfers = len(driver_out) + len(constr_out)
    penalty_pts = max(0, n_transfers - max_free_transfers) * penalty_per_transfer

    return {
        "drivers": selected_drivers,
        "turbo_driver": turbo_driver,
        "constructors": selected_constructors,
        "total_price": round(spent, 1),
        "remaining_budget": round(budget - spent, 1),
        "gross_points": round(gross_pts, 2),
        "budget_value": round(budget_value, 2),
        "penalty_pts": penalty_pts,
        "total_points": round(gross_pts - penalty_pts, 2),
        "n_transfers": n_transfers,
        "driver_transfers_out": driver_out,
        "driver_transfers_in":  driver_in,
        "constructor_transfers_out": constr_out,
        "constructor_transfers_in":  constr_in,
    }


def solve_portfolio_transfers(
    df: pd.DataFrame,
    current_teams: list[dict],
    budgets: list[float],
    max_pairwise_overlap: int = 5,
    locked: list[str] | None = None,
    banned: list[str] | None = None,
    max_free_transfers: int = 2,
    penalty_per_transfer: int = 10,
    budget_pts_weight: float = 0.0,
    team_weights: list[float] | None = None,
) -> list[dict]:
    """
    Solve optimal transfers for all teams jointly in a single ILP.

    All teams are optimised simultaneously with pairwise overlap constraints
    enforced across every team pair at once. Unlike a sequential approach, no
    team is prioritised over others — the solver trades off freely between teams
    to maximise the weighted total portfolio score.

    Parameters
    ----------
    current_teams : list of enrich_team() dicts from fetch_teams.py
    budgets       : available budget per team (total_price + budget_remaining)
    team_weights  : per-team objective weights (e.g. [0.2, 0.4, 0.4]).
                    Higher weight = solver prioritises that team more.
                    Defaults to equal weights [1.0, 1.0, ...] if None.
    """
    n_teams = len(current_teams)
    locked  = {c.upper() for c in (locked or [])}
    banned  = {c.upper() for c in (banned or [])}

    if team_weights is None:
        team_weights = [1.0] * n_teams
    if len(team_weights) != n_teams:
        raise ValueError(f"team_weights must have {n_teams} entries, got {len(team_weights)}")

    overlap = locked & banned
    if overlap:
        raise ValueError(f"Codes appear in both locked and banned: {overlap}")

    drivers      = df[df["type"] == "driver"].reset_index(drop=True)
    constructors = df[df["type"] == "constructor"].reset_index(drop=True)
    n_d, n_c     = len(drivers), len(constructors)

    all_codes = set(df["name"].str.upper())
    unknown   = (locked | banned) - all_codes
    if unknown:
        raise ValueError(f"Codes not found in data: {unknown}")
    if n_d < 5:
        raise ValueError(f"Need at least 5 drivers, got {n_d}.")
    if n_c < 2:
        raise ValueError(f"Need at least 2 constructors, got {n_c}.")

    prob      = pulp.LpProblem("F1_Fantasy_Joint_Portfolio", pulp.LpMaximize)
    has_delta = "xDeltaPrice" in df.columns and budget_pts_weight != 0.0

    # ── Decision variables (one set per team) ─────────────────────────────────
    x_d = [[pulp.LpVariable(f"x_d_{k}_{i}", cat="Binary") for i in range(n_d)] for k in range(n_teams)]
    x_c = [[pulp.LpVariable(f"x_c_{k}_{j}", cat="Binary") for j in range(n_c)] for k in range(n_teams)]
    t_d = [[pulp.LpVariable(f"t_d_{k}_{i}", cat="Binary") for i in range(n_d)] for k in range(n_teams)]
    e   =  [pulp.LpVariable(f"e_{k}", lowBound=0, cat="Integer")               for k in range(n_teams)]

    # ── Objective: weighted sum of (score − penalty) across all teams ─────────
    obj_terms = []
    for k in range(n_teams):
        w     = team_weights[k]
        score = pulp.lpSum(
            (x_d[k][i] + t_d[k][i]) * drivers.loc[i, "expected_points"] for i in range(n_d)
        ) + pulp.lpSum(
            x_c[k][j] * constructors.loc[j, "expected_points"] for j in range(n_c)
        )
        if has_delta:
            score += pulp.lpSum(
                x_d[k][i] * drivers.loc[i, "xDeltaPrice"] * budget_pts_weight for i in range(n_d)
            ) + pulp.lpSum(
                x_c[k][j] * constructors.loc[j, "xDeltaPrice"] * budget_pts_weight for j in range(n_c)
            )
        obj_terms.append(w * (score - penalty_per_transfer * e[k]))
    prob += pulp.lpSum(obj_terms)

    # ── Per-team standard + transfer constraints ───────────────────────────────
    for k, (team, budget) in enumerate(zip(current_teams, budgets)):
        prob += pulp.lpSum(x_d[k]) == 5, f"k{k}_5_drivers"
        prob += pulp.lpSum(x_c[k]) == 2, f"k{k}_2_constructors"
        prob += pulp.lpSum(t_d[k]) == 1, f"k{k}_1_turbo"

        for i in range(n_d):
            prob += t_d[k][i] <= x_d[k][i], f"k{k}_turbo_sel_{i}"

        for i in range(n_d):
            code = drivers.loc[i, "name"].upper()
            if code in locked:
                prob += x_d[k][i] == 1, f"k{k}_lock_d_{code}"
            elif code in banned:
                prob += x_d[k][i] == 0, f"k{k}_ban_d_{code}"

        for j in range(n_c):
            code = constructors.loc[j, "name"].upper()
            if code in locked:
                prob += x_c[k][j] == 1, f"k{k}_lock_c_{code}"
            elif code in banned:
                prob += x_c[k][j] == 0, f"k{k}_ban_c_{code}"

        prob += (
            pulp.lpSum(x_d[k][i] * drivers.loc[i, "price"] for i in range(n_d))
            + pulp.lpSum(x_c[k][j] * constructors.loc[j, "price"] for j in range(n_c))
            <= budget,
            f"k{k}_budget",
        )

        current_d = {d["tla"].upper() for d in team["drivers"]}
        current_c = {c["tla"].upper() for c in team["constructors"]}
        kept = (
            pulp.lpSum(x_d[k][i] for i in range(n_d) if drivers.loc[i, "name"].upper() in current_d)
            + pulp.lpSum(x_c[k][j] for j in range(n_c) if constructors.loc[j, "name"].upper() in current_c)
        )
        prob += e[k] >= 7 - kept - max_free_transfers, f"k{k}_excess_lb"

    # ── Pairwise overlap constraints (all pairs, jointly) ─────────────────────
    # Overlap between teams a and b = shared driver picks + shared constructor
    # picks + same turbo. Requires auxiliary binary vars to linearise AND of
    # two binary variables: z = x_a AND x_b  ⟺  z≤x_a, z≤x_b, z≥x_a+x_b-1
    for a, b in [(a, b) for a in range(n_teams) for b in range(a + 1, n_teams)]:
        z_d = [pulp.LpVariable(f"z_d_{a}{b}_{i}", cat="Binary") for i in range(n_d)]
        z_c = [pulp.LpVariable(f"z_c_{a}{b}_{j}", cat="Binary") for j in range(n_c)]
        w_t = [pulp.LpVariable(f"w_t_{a}{b}_{i}", cat="Binary") for i in range(n_d)]

        for i in range(n_d):
            prob += z_d[i] <= x_d[a][i],                  f"zd_{a}{b}_{i}_ub_a"
            prob += z_d[i] <= x_d[b][i],                  f"zd_{a}{b}_{i}_ub_b"
            prob += z_d[i] >= x_d[a][i] + x_d[b][i] - 1, f"zd_{a}{b}_{i}_lb"
            prob += w_t[i] <= t_d[a][i],                  f"wt_{a}{b}_{i}_ub_a"
            prob += w_t[i] <= t_d[b][i],                  f"wt_{a}{b}_{i}_ub_b"
            prob += w_t[i] >= t_d[a][i] + t_d[b][i] - 1, f"wt_{a}{b}_{i}_lb"

        for j in range(n_c):
            prob += z_c[j] <= x_c[a][j],                  f"zc_{a}{b}_{j}_ub_a"
            prob += z_c[j] <= x_c[b][j],                  f"zc_{a}{b}_{j}_ub_b"
            prob += z_c[j] >= x_c[a][j] + x_c[b][j] - 1, f"zc_{a}{b}_{j}_lb"

        prob += (
            pulp.lpSum(z_d) + pulp.lpSum(z_c) + pulp.lpSum(w_t) <= max_pairwise_overlap,
            f"overlap_{a}{b}",
        )

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        raise RuntimeError(
            f"Solver did not find an optimal solution. Status: {pulp.LpStatus[prob.status]}"
        )

    # ── Parse one result dict per team ────────────────────────────────────────
    has_delta_d = "xDeltaPrice" in drivers.columns
    has_delta_c = "xDeltaPrice" in constructors.columns
    results     = []

    for k, (team, budget) in enumerate(zip(current_teams, budgets)):
        current_d = {d["tla"].upper() for d in team["drivers"]}
        current_c = {c["tla"].upper() for c in team["constructors"]}

        sel_drivers, new_d_tlas, turbo_driver = [], set(), None
        for i in range(n_d):
            if pulp.value(x_d[k][i]) > 0.5:
                row      = drivers.loc[i]
                is_turbo = pulp.value(t_d[k][i]) > 0.5
                sel_drivers.append({
                    "name":            row["name"],
                    "price":           row["price"],
                    "expected_points": row["expected_points"],
                    "xDeltaPrice":     float(row["xDeltaPrice"]) if has_delta_d else 0.0,
                    "is_turbo":        is_turbo,
                })
                new_d_tlas.add(row["name"].upper())
                if is_turbo:
                    turbo_driver = row["name"]

        sel_constructors, new_c_tlas = [], set()
        for j in range(n_c):
            if pulp.value(x_c[k][j]) > 0.5:
                row = constructors.loc[j]
                sel_constructors.append({
                    "name":            row["name"],
                    "price":           row["price"],
                    "expected_points": row["expected_points"],
                    "xDeltaPrice":     float(row["xDeltaPrice"]) if has_delta_c else 0.0,
                })
                new_c_tlas.add(row["name"].upper())

        sel_drivers.sort(key=lambda d: d["is_turbo"])

        spent     = sum(d["price"] for d in sel_drivers) + sum(c["price"] for c in sel_constructors)
        gross_pts = sum(
            (d["expected_points"] * 2 if d["is_turbo"] else d["expected_points"])
            for d in sel_drivers
        ) + sum(c["expected_points"] for c in sel_constructors)
        budget_value = (
            sum(d["xDeltaPrice"] for d in sel_drivers)
            + sum(c["xDeltaPrice"] for c in sel_constructors)
        ) * budget_pts_weight

        driver_out  = sorted(current_d - new_d_tlas)
        driver_in   = sorted(new_d_tlas - current_d)
        constr_out  = sorted(current_c - new_c_tlas)
        constr_in   = sorted(new_c_tlas - current_c)
        n_transfers = len(driver_out) + len(constr_out)
        penalty_pts = max(0, n_transfers - max_free_transfers) * penalty_per_transfer

        results.append({
            "drivers":                   sel_drivers,
            "turbo_driver":              turbo_driver,
            "constructors":              sel_constructors,
            "total_price":               round(spent, 1),
            "remaining_budget":          round(budget - spent, 1),
            "gross_points":              round(gross_pts, 2),
            "budget_value":              round(budget_value, 2),
            "penalty_pts":               penalty_pts,
            "total_points":              round(gross_pts - penalty_pts, 2),
            "n_transfers":               n_transfers,
            "driver_transfers_out":      driver_out,
            "driver_transfers_in":       driver_in,
            "constructor_transfers_out": constr_out,
            "constructor_transfers_in":  constr_in,
        })

    return results


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(
    result: dict,
    budget: float,
    rank: int = 1,
    total: int = 1,
    budget_pts_weight: float = 0.0,
) -> None:
    sep = "─" * 62
    show_delta = budget_pts_weight != 0.0 and result.get("budget_value", 0.0) != 0.0

    if total > 1:
        title = f"RANK #{rank}  —  {result['total_points']:.2f} pts"
    else:
        title = "F1 FANTASY OPTIMAL TEAM"
    print(f"\n{title:^62}")
    print(sep)

    print(f"\n{'DRIVERS':}")
    if show_delta:
        print(f"  {'Name':<25} {'Price':>6}  {'xPts':>6}  {'ΔPrice':>7}  {'Note'}")
        print(f"  {'─'*25} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*13}")
    else:
        print(f"  {'Name':<25} {'Price':>6}  {'Pts':>6}  {'Note'}")
        print(f"  {'─'*25} {'─'*6}  {'─'*6}  {'─'*13}")
    for d in result["drivers"]:
        tag = " ★ TURBO (2x)" if d["is_turbo"] else ""
        pts_disp = f"{d['expected_points']*2:.1f}" if d["is_turbo"] else f"{d['expected_points']:.1f}"
        raw = f"({d['expected_points']:.1f})" if d["is_turbo"] else ""
        if show_delta:
            delta = d.get("xDeltaPrice", 0.0)
            delta_str = f"{delta:+.2f}"
            print(f"  {d['name']:<25} {d['price']:>6.1f}  {pts_disp:>6}{raw:>4}  {delta_str:>7}  {tag}")
        else:
            print(f"  {d['name']:<25} {d['price']:>6.1f}  {pts_disp:>6}{raw:>8}{tag}")

    print(f"\n{'CONSTRUCTORS':}")
    if show_delta:
        print(f"  {'Name':<25} {'Price':>6}  {'xPts':>6}  {'ΔPrice':>7}")
        print(f"  {'─'*25} {'─'*6}  {'─'*6}  {'─'*7}")
    else:
        print(f"  {'Name':<25} {'Price':>6}  {'Pts':>6}")
        print(f"  {'─'*25} {'─'*6}  {'─'*6}")
    for c in result["constructors"]:
        if show_delta:
            delta = c.get("xDeltaPrice", 0.0)
            delta_str = f"{delta:+.2f}"
            print(f"  {c['name']:<25} {c['price']:>6.1f}  {c['expected_points']:>6.1f}  {delta_str:>7}")
        else:
            print(f"  {c['name']:<25} {c['price']:>6.1f}  {c['expected_points']:>6.1f}")

    print(f"\n{sep}")
    print(f"  Budget:           {budget:.1f}")
    print(f"  Total spent:      {result['total_price']:.1f}")
    print(f"  Remaining:        {result['remaining_budget']:.1f}")
    print(f"  Race xPts:        {result['total_points']:.2f}  (incl. turbo bonus)")
    if show_delta:
        bv = result.get("budget_value", 0.0)
        print(f"  Budget value:     {bv:+.2f}  (xDeltaPrice × {budget_pts_weight:.1f} pts/M)")
        print(f"  Combined score:   {result['total_points'] + bv:.2f}")
    print(sep)


def print_portfolio(
    teams: list[dict],
    budget: float,
    max_pairwise_overlap: int,
    budget_pts_weight: float = 0.0,
) -> None:
    """Print all portfolio teams, then a pairwise overlap + xPts-cost summary."""
    optimal_pts = teams[0]["total_points"] if teams else 0.0

    for rank, team in enumerate(teams, start=1):
        print_result(team, budget=budget, rank=rank, total=len(teams), budget_pts_weight=budget_pts_weight)

    if len(teams) < 2:
        return

    show_combined = budget_pts_weight != 0.0
    optimal_combined = optimal_pts + (teams[0].get("budget_value", 0.0) if teams else 0.0)

    sep = "─" * 54
    print(f"\n{'PORTFOLIO SUMMARY':^54}")
    print(sep)
    print(f"  Max pairwise overlap allowed : {max_pairwise_overlap} / 8")
    print(f"  Optimal (Team 1) xPts       : {optimal_pts:.2f}")
    if show_combined:
        print(f"  Optimal (Team 1) Combined   : {optimal_combined:.2f}")
    print()

    # xPts cost per team
    if show_combined:
        print(f"  {'Team':<8} {'xPts':>8}  {'Combined':>10}  {'vs Optimal':>12}")
        print(f"  {'─'*8} {'─'*8}  {'─'*10}  {'─'*12}")
        for i, team in enumerate(teams, start=1):
            combined = team["total_points"] + team.get("budget_value", 0.0)
            cost = combined - optimal_combined
            cost_str = f"{cost:+.2f}"
            print(f"  Team {i:<3}  {team['total_points']:>8.2f}  {combined:>10.2f}  {cost_str:>12}")
    else:
        print(f"  {'Team':<8} {'xPts':>8}  {'vs Optimal':>12}")
        print(f"  {'─'*8} {'─'*8}  {'─'*12}")
        for i, team in enumerate(teams, start=1):
            cost = team["total_points"] - optimal_pts
            cost_str = f"{cost:+.2f}"
            print(f"  Team {i:<3}  {team['total_points']:>8.2f}  {cost_str:>12}")

    # Pairwise overlap matrix
    print()
    print(f"  Pairwise overlap (out of 8 decisions):")
    header = f"  {'':8}" + "".join(f"  T{j+1}" for j in range(len(teams)))
    print(header)
    for i, t1 in enumerate(teams):
        row = f"  Team {i+1:<3}"
        for j, t2 in enumerate(teams):
            if j <= i:
                row += "    —"
            else:
                ov = compute_overlap(t1, t2)
                row += f"  {ov:>3}"
        print(row)

    # Shared picks breakdown per pair
    print()
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            d1 = {d["name"].upper() for d in t1["drivers"]}
            d2 = {d["name"].upper() for d in t2["drivers"]}
            c1 = {c["name"].upper() for c in t1["constructors"]}
            c2 = {c["name"].upper() for c in t2["constructors"]}
            shared_d = sorted(d1 & d2)
            shared_c = sorted(c1 & c2)
            same_turbo = t1["turbo_driver"].upper() == t2["turbo_driver"].upper()
            print(f"  Team {i+1} vs Team {j+1}:")
            print(f"    Shared drivers      : {', '.join(shared_d) if shared_d else 'none'}")
            print(f"    Shared constructors : {', '.join(shared_c) if shared_c else 'none'}")
            print(f"    Same turbo          : {'yes (' + t1['turbo_driver'] + ')' if same_turbo else 'no'}")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="F1 Fantasy ILP solver — maximise expected points within budget."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=settings["data_file"],
        help=f"Path to CSV file (default from settings.json: {settings['data_file']})",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help=f"Total budget cap (default from settings.json: {settings['budget']})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help=f"Number of top solutions to return (default from settings.json: {settings['top_n']})",
    )
    parser.add_argument(
        "--portfolio",
        type=int,
        nargs="?",
        const=settings["portfolio_n"] or 3,
        default=None,
        metavar="N",
        help=(
            f"Solve a diversified portfolio of N teams instead of top-N mode "
            f"(default N from settings.json: {settings['portfolio_n']})"
        ),
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        metavar="K",
        help=(
            f"Max pairwise overlap (0–8) when using --portfolio "
            f"(default from settings.json: {settings['portfolio_max_overlap']}). "
            "Overlap counts: 5 driver picks + 2 constructor picks + 1 turbo = 8 max."
        ),
    )
    parser.add_argument(
        "--pts-per-1m",
        type=float,
        default=None,
        metavar="PTS",
        help=(
            f"Points earned per 1M budget increase per future race "
            f"(default from settings.json: {settings['pts_per_1m_per_race']}). "
            "Set to 0 to disable budget optimisation."
        ),
    )
    parser.add_argument(
        "--remaining-races",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Number of remaining races to value budget gains over "
            f"(default from settings.json: {settings['remaining_races']})."
        ),
    )
    args = parser.parse_args()

    # CLI args override settings; settings override hard-coded defaults
    budget = args.budget if args.budget is not None else settings["budget"]
    top_n = args.top if args.top is not None else settings["top_n"]
    # portfolio mode: CLI flag > settings.json portfolio_n > None (top-N mode)
    portfolio_n = args.portfolio if args.portfolio is not None else settings["portfolio_n"]
    max_overlap = args.overlap if args.overlap is not None else settings["portfolio_max_overlap"]
    pts_per_1m = args.pts_per_1m if args.pts_per_1m is not None else settings["pts_per_1m_per_race"]
    remaining_races = args.remaining_races if args.remaining_races is not None else settings["remaining_races"]
    budget_pts_weight = pts_per_1m * remaining_races

    locked = [c.upper() for c in settings.get("locked", [])]
    banned = [c.upper() for c in settings.get("banned", [])]

    print(f"Settings loaded from: {SETTINGS_FILE}")
    print(f"  data_file : {args.csv_file}")
    print(f"  budget    : {budget}")
    if portfolio_n is not None:
        print(f"  mode      : portfolio ({portfolio_n} teams, max overlap {max_overlap}/8)")
    else:
        print(f"  top_n     : {top_n}")
    if budget_pts_weight != 0.0:
        print(f"  pts/1M/race : {pts_per_1m}  ×  {remaining_races} races  =  {budget_pts_weight:.1f} pts/M total")
    if locked:
        print(f"  locked    : {', '.join(locked)}")
    if banned:
        print(f"  banned    : {', '.join(banned)}")

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: file '{args.csv_file}' not found.")
        sys.exit(1)

    # Normalise alternative column names
    col_aliases = {"code": "name", "xPts": "expected_points"}
    df = df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns})

    required_cols = {"name", "type", "price", "expected_points"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: CSV is missing columns: {missing}")
        sys.exit(1)

    df["type"] = df["type"].str.lower().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["expected_points"] = pd.to_numeric(df["expected_points"], errors="coerce")
    if "xDeltaPrice" in df.columns:
        df["xDeltaPrice"] = pd.to_numeric(df["xDeltaPrice"], errors="coerce").fillna(0.0)

    try:
        if portfolio_n is not None:
            results = solve_portfolio(
                df,
                budget=budget,
                locked=locked,
                banned=banned,
                n_teams=portfolio_n,
                max_pairwise_overlap=max_overlap,
                budget_pts_weight=budget_pts_weight,
            )
        else:
            results = solve_top_n(
                df, budget=budget, locked=locked, banned=banned, n=top_n,
                budget_pts_weight=budget_pts_weight,
            )
    except ValueError as e:
        print(f"Solver error: {e}")
        sys.exit(1)

    if not results:
        print("No feasible solution found within the given constraints.")
        sys.exit(1)

    if portfolio_n is not None:
        print_portfolio(results, budget=budget, max_pairwise_overlap=max_overlap, budget_pts_weight=budget_pts_weight)
    else:
        for rank, result in enumerate(results, start=1):
            print_result(result, budget=budget, rank=rank, total=len(results), budget_pts_weight=budget_pts_weight)


if __name__ == "__main__":
    main()
