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
DEFAULT_SETTINGS = {"budget": 100.0, "data_file": "sample_data.csv", "locked": [], "banned": [], "top_n": 5}


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
) -> dict:
    """
    Run the F1 Fantasy ILP optimiser.

    Parameters
    ----------
    df         : DataFrame with columns [name, type, price, expected_points]
    budget     : Total spend cap (default 100)
    locked     : List of codes that MUST be in the team
    banned     : List of codes that MUST NOT be in the team
    exclusions : List of previous result dicts whose exact team+turbo combo must not repeat
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
    # Turbo driver earns 2x points → regular pts + bonus pts
    # Total = Σ (x_d[i] + t_d[i]) * pts_i  +  Σ x_c[j] * pts_j
    driver_pts = pulp.lpSum(
        (x_d[i] + t_d[i]) * drivers.loc[i, "expected_points"] for i in range(n_d)
    )
    constructor_pts = pulp.lpSum(
        x_c[j] * constructors.loc[j, "expected_points"] for j in range(n_c)
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

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    if prob.status != 1:  # 1 = Optimal
        raise RuntimeError(f"Solver did not find an optimal solution. Status: {status}")

    # ── Parse results ─────────────────────────────────────────────────────────
    selected_drivers = []
    turbo_driver = None

    for i in range(n_d):
        if pulp.value(x_d[i]) > 0.5:
            row = drivers.loc[i]
            is_turbo = pulp.value(t_d[i]) > 0.5
            selected_drivers.append({
                "name": row["name"],
                "price": row["price"],
                "expected_points": row["expected_points"],
                "is_turbo": is_turbo,
            })
            if is_turbo:
                turbo_driver = row["name"]

    selected_constructors = []
    for j in range(n_c):
        if pulp.value(x_c[j]) > 0.5:
            row = constructors.loc[j]
            selected_constructors.append({
                "name": row["name"],
                "price": row["price"],
                "expected_points": row["expected_points"],
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

    return {
        "drivers": selected_drivers,
        "turbo_driver": turbo_driver,
        "constructors": selected_constructors,
        "total_price": round(spent, 1),
        "remaining_budget": round(budget - spent, 1),
        "total_points": round(pts, 2),
    }


# ── Top-N solver ──────────────────────────────────────────────────────────────

def solve_top_n(
    df: pd.DataFrame,
    budget: float = 100.0,
    locked: list[str] | None = None,
    banned: list[str] | None = None,
    n: int = 5,
) -> list[dict]:
    """Return the top-n distinct optimal teams, ranked by total points."""
    results = []
    for _ in range(n):
        try:
            result = solve(df, budget=budget, locked=locked, banned=banned, exclusions=results)
        except RuntimeError:
            break
        results.append(result)
    return results


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(result: dict, budget: float, rank: int = 1, total: int = 1) -> None:
    sep = "─" * 54

    if total > 1:
        title = f"RANK #{rank}  —  {result['total_points']:.2f} pts"
    else:
        title = "F1 FANTASY OPTIMAL TEAM"
    print(f"\n{title:^54}")
    print(sep)

    print(f"\n{'DRIVERS':}")
    print(f"  {'Name':<25} {'Price':>6}  {'Pts':>6}  {'Note'}")
    print(f"  {'─'*25} {'─'*6}  {'─'*6}  {'─'*10}")
    for d in result["drivers"]:
        tag = " ★ TURBO (2x)" if d["is_turbo"] else ""
        pts_disp = f"{d['expected_points']*2:.1f}" if d["is_turbo"] else f"{d['expected_points']:.1f}"
        raw = f"({d['expected_points']:.1f})" if d["is_turbo"] else ""
        print(f"  {d['name']:<25} {d['price']:>6.1f}  {pts_disp:>6}{raw:>8}{tag}")

    print(f"\n{'CONSTRUCTORS':}")
    print(f"  {'Name':<25} {'Price':>6}  {'Pts':>6}")
    print(f"  {'─'*25} {'─'*6}  {'─'*6}")
    for c in result["constructors"]:
        print(f"  {c['name']:<25} {c['price']:>6.1f}  {c['expected_points']:>6.1f}")

    print(f"\n{sep}")
    print(f"  Budget:           {budget:.1f}")
    print(f"  Total spent:      {result['total_price']:.1f}")
    print(f"  Remaining:        {result['remaining_budget']:.1f}")
    print(f"  Total points:     {result['total_points']:.2f}  (incl. turbo bonus)")
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
    args = parser.parse_args()

    # CLI args override settings; settings override hard-coded defaults
    budget = args.budget if args.budget is not None else settings["budget"]
    top_n = args.top if args.top is not None else settings["top_n"]

    locked = [c.upper() for c in settings.get("locked", [])]
    banned = [c.upper() for c in settings.get("banned", [])]

    print(f"Settings loaded from: {SETTINGS_FILE}")
    print(f"  data_file : {args.csv_file}")
    print(f"  budget    : {budget}")
    print(f"  top_n     : {top_n}")
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

    try:
        results = solve_top_n(df, budget=budget, locked=locked, banned=banned, n=top_n)
    except ValueError as e:
        print(f"Solver error: {e}")
        sys.exit(1)

    if not results:
        print("No feasible solution found within the given constraints.")
        sys.exit(1)

    for rank, result in enumerate(results, start=1):
        print_result(result, budget=budget, rank=rank, total=len(results))


if __name__ == "__main__":
    main()
