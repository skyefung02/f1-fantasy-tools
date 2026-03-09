"""
F1 Fantasy Transfer Advisor
============================
Fetches your 3 current teams from the F1 Fantasy website and recommends
optimal transfers for the next race, balancing expected points and
portfolio diversity.

Usage
-----
    python transfer_advisor.py
    python transfer_advisor.py --overlap 4
    python transfer_advisor.py data.csv --overlap 6
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import requests

from fetch_teams import F1FantasyClient, enrich_team, load_cookie, load_settings
from solver import solve_portfolio_transfers, compute_overlap


# ── Helpers ───────────────────────────────────────────────────────────────────

def team_budget(team: dict) -> float:
    """Total spendable budget = current team value + remaining cash."""
    return round(team["total_price"] + team["budget_remaining"], 1)


# ── Display ───────────────────────────────────────────────────────────────────

def print_transfer_plan(
    result: dict,
    current_team: dict,
    budget: float,
    team_idx: int,
    budget_pts_weight: float = 0.0,
    xdelta_confidence: float = 1.0,
) -> None:
    sep = "─" * 60
    wide = "=" * 60

    print(f"\n{wide}")
    print(f"  TEAM {team_idx}  —  {current_team['name']}")
    print(f"  Budget available: {budget:.1f}m")
    print(wide)

    # ── Transfers summary ────────────────────────────────────────────────────
    n = result["n_transfers"]
    free = 2
    extra = max(0, n - free)
    if n == 0:
        print("\n  No changes — current team is already optimal.")
    else:
        penalty_note = f"  (-{result['penalty_pts']:.0f} pts penalty)" if extra else "  (all free)"
        print(f"\n  TRANSFERS: {n}{penalty_note}")
        print(f"  {'─'*56}")

        for tla in result["driver_transfers_out"]:
            print(f"    OUT  (driver)     : {tla}")
        for tla in result["driver_transfers_in"]:
            print(f"     IN  (driver)     : {tla}")

        for tla in result["constructor_transfers_out"]:
            print(f"    OUT  (constructor): {tla}")
        for tla in result["constructor_transfers_in"]:
            print(f"     IN  (constructor): {tla}")

    # ── New team ─────────────────────────────────────────────────────────────
    new_picks = (
        set(result["driver_transfers_in"])
        | set(result["constructor_transfers_in"])
    )

    show_delta = budget_pts_weight != 0.0

    print(f"\n  NEW TEAM")
    if show_delta:
        print(f"  {'TLA':<8} {'Price':>6}  {'xPts':>6}  {'ΔPrice':>7}  {'Note'}")
        print(f"  {'─'*8} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*20}")
    else:
        print(f"  {'TLA':<8} {'Price':>6}  {'xPts':>6}  {'Note'}")
        print(f"  {'─'*8} {'─'*6}  {'─'*6}  {'─'*20}")

    for d in result["drivers"]:
        tla = d["name"].upper()
        tag = " ★ TURBO (2x)" if d["is_turbo"] else ""
        new_tag = "  ← NEW" if tla in new_picks else ""
        pts_disp = f"{d['expected_points'] * 2:.1f}" if d["is_turbo"] else f"{d['expected_points']:.1f}"
        if show_delta:
            delta_str = f"{d.get('xDeltaPrice', 0.0):+.2f}"
            print(f"  {tla:<8} {d['price']:>6.1f}  {pts_disp:>6}  {delta_str:>7}  {tag}{new_tag}")
        else:
            print(f"  {tla:<8} {d['price']:>6.1f}  {pts_disp:>6}  {tag}{new_tag}")

    for c in result["constructors"]:
        tla = c["name"].upper()
        new_tag = "  ← NEW" if tla in new_picks else ""
        if show_delta:
            delta_str = f"{c.get('xDeltaPrice', 0.0):+.2f}"
            print(f"  {tla:<8} {c['price']:>6.1f}  {c['expected_points']:>6.1f}  {delta_str:>7}  {new_tag}")
        else:
            print(f"  {tla:<8} {c['price']:>6.1f}  {c['expected_points']:>6.1f}  {new_tag}")

    # ── Points breakdown ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  Gross xPts   : {result['gross_points']:.2f}")
    if result["penalty_pts"]:
        print(f"  Penalty      : -{result['penalty_pts']:.0f}  ({extra} extra transfer(s) × 10 pts)")
    print(f"  Net xPts     : {result['total_points']:.2f}")
    if show_delta:
        bv = result.get("budget_value", 0.0) * xdelta_confidence
        conf_note = f" × {xdelta_confidence:.0%} confidence" if xdelta_confidence != 1.0 else ""
        print(f"  Budget value : {bv:+.2f}  (xDeltaPrice × {budget_pts_weight:.1f} pts/M{conf_note})")
        print(f"  Combined     : {result['total_points'] + bv:.2f}")
    print(f"  Spent        : {result['total_price']:.1f}m  (remaining: {result['remaining_budget']:.1f}m)")
    print(sep)


def print_portfolio_summary(
    results: list[dict],
    current_teams: list[dict],
    max_pairwise_overlap: int,
    budget_pts_weight: float = 0.0,
    xdelta_confidence: float = 1.0,
) -> None:
    if len(results) < 2:
        return

    show_combined = budget_pts_weight != 0.0

    sep = "─" * 60
    print(f"\n{'PORTFOLIO SUMMARY':^60}")
    print(sep)
    print(f"  Max pairwise overlap : {max_pairwise_overlap} / 8")
    print()

    # Per-team stats
    if show_combined:
        print(f"  {'Team':<10} {'Net xPts':>10}  {'Combined':>10}  {'Transfers':>10}  {'Penalty':>8}")
        print(f"  {'─'*10} {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")
        for i, (r, t) in enumerate(zip(results, current_teams), 1):
            combined = r['total_points'] + r.get('budget_value', 0.0) * xdelta_confidence
            print(
                f"  Team {i:<5}  {r['total_points']:>10.2f}"
                f"  {combined:>10.2f}"
                f"  {r['n_transfers']:>10}"
                f"  {r['penalty_pts']:>8.0f}"
            )
    else:
        print(f"  {'Team':<10} {'Net xPts':>10}  {'Transfers':>10}  {'Penalty':>8}")
        print(f"  {'─'*10} {'─'*10}  {'─'*10}  {'─'*8}")
        for i, (r, t) in enumerate(zip(results, current_teams), 1):
            print(
                f"  Team {i:<5}  {r['total_points']:>10.2f}"
                f"  {r['n_transfers']:>10}"
                f"  {r['penalty_pts']:>8.0f}"
            )

    # Pairwise overlap matrix
    print()
    print(f"  Pairwise overlap (out of 8 decisions):")
    header = f"  {'':10}" + "".join(f"  T{j+1}" for j in range(len(results)))
    print(header)
    for i, t1 in enumerate(results):
        row = f"  Team {i+1:<5}"
        for j, t2 in enumerate(results):
            if j <= i:
                row += "    —"
            else:
                ov = compute_overlap(t1, t2)
                row += f"  {ov:>3}"
        print(row)

    # Shared picks breakdown per pair
    print()
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            t1, t2 = results[i], results[j]
            d1 = {d["name"].upper() for d in t1["drivers"]}
            d2 = {d["name"].upper() for d in t2["drivers"]}
            c1 = {c["name"].upper() for c in t1["constructors"]}
            c2 = {c["name"].upper() for c in t2["constructors"]}
            shared_d = sorted(d1 & d2)
            shared_c = sorted(c1 & c2)
            same_turbo = t1["turbo_driver"].upper() == t2["turbo_driver"].upper()
            print(f"  Team {i+1} vs Team {j+1}:")
            print(f"    Shared drivers      : {', '.join(shared_d) or 'none'}")
            print(f"    Shared constructors : {', '.join(shared_c) or 'none'}")
            print(f"    Same turbo          : {'yes (' + t1['turbo_driver'] + ')' if same_turbo else 'no'}")

    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description=(
            "F1 Fantasy Transfer Advisor — recommends optimal transfers "
            "across your 3 teams for the next race."
        )
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=settings.get("data_file", "sample_data.csv"),
        help="Path to xPts CSV file (default from settings.json)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=settings.get("portfolio_max_overlap", 5),
        metavar="K",
        help=(
            "Max pairwise overlap between the 3 resulting teams (0–8, default 5). "
            "Overlap counts: 5 driver picks + 2 constructor picks + 1 turbo = 8 max."
        ),
    )
    parser.add_argument(
        "--pts-per-1m",
        type=float,
        default=settings.get("pts_per_1m_per_race", 0.0),
        metavar="PTS",
        help=(
            "Points earned per 1M budget increase per future race "
            "(default from settings.json). Set to 0 to disable budget optimisation."
        ),
    )
    parser.add_argument(
        "--remaining-races",
        type=int,
        default=settings.get("remaining_races", 23),
        metavar="N",
        help="Number of remaining races to value budget gains over (default from settings.json).",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=settings.get("team_weights", None),
        metavar="W",
        help=(
            "Per-team objective weights, one per team (e.g. --weights 0.2 0.4 0.4). "
            "Higher weight = solver prioritises that team more. "
            "Defaults to equal weighting. Also settable via 'team_weights' in settings.json."
        ),
    )
    args = parser.parse_args()

    # ── Load xPts data ────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: file '{args.csv_file}' not found.")
        sys.exit(1)

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

    budget_pts_weight = args.pts_per_1m * args.remaining_races

    # ── Fetch current teams ───────────────────────────────────────────────────
    cookie = load_cookie()
    if not cookie:
        print("Error: F1_FANTASY_COOKIE not set. See README for setup instructions.")
        sys.exit(1)

    user_uuid = settings.get("user_uuid")
    if not user_uuid:
        print("Error: 'user_uuid' not set in settings.json.")
        sys.exit(1)

    gameday = settings.get("gameday", 1)
    client = F1FantasyClient(cookie=cookie)

    print(f"Fetching player data (gameday {gameday})...")
    try:
        players = client.get_players(gameday=gameday)
    except requests.HTTPError as e:
        print(f"Error fetching players: {e}")
        sys.exit(1)

    print("Fetching your teams...")
    try:
        raw_teams = client.get_my_teams(user_uuid=user_uuid, gameday=gameday)
    except requests.HTTPError as e:
        print(f"Error fetching teams: {e}")
        print("Your cookie may be expired — re-extract it from the browser.")
        sys.exit(1)

    if not raw_teams:
        print("No teams found. Check your cookie and user_uuid.")
        sys.exit(0)

    current_teams = [enrich_team(raw, players) for raw in raw_teams]
    budgets = [team_budget(t) for t in current_teams]

    locked = [c.upper() for c in settings.get("locked", [])]
    banned = [c.upper() for c in settings.get("banned", [])]

    team_weights      = args.weights
    xdelta_confidence = float(settings.get("xdelta_confidence", 1.0))

    if budget_pts_weight != 0.0:
        print(f"  pts/1M/race : {args.pts_per_1m}  ×  {args.remaining_races} races  =  {budget_pts_weight:.1f} pts/M total")
        if xdelta_confidence != 1.0:
            print(f"  xdelta conf : {xdelta_confidence:.0%}  (budget value discounted in reported Combined)")
    if team_weights:
        print(f"  team weights: {team_weights}")
    print(f"\nSolving transfers (max overlap {args.overlap}/8)...\n")
    try:
        results = solve_portfolio_transfers(
            df,
            current_teams=current_teams,
            budgets=budgets,
            max_pairwise_overlap=args.overlap,
            locked=locked,
            banned=banned,
            budget_pts_weight=budget_pts_weight,
            team_weights=team_weights,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Solver error: {e}")
        sys.exit(1)

    for i, (result, current_team, budget) in enumerate(
        zip(results, current_teams, budgets), start=1
    ):
        print_transfer_plan(result, current_team, budget, team_idx=i, budget_pts_weight=budget_pts_weight, xdelta_confidence=xdelta_confidence)

    print_portfolio_summary(results, current_teams, max_pairwise_overlap=args.overlap, budget_pts_weight=budget_pts_weight, xdelta_confidence=xdelta_confidence)


if __name__ == "__main__":
    main()
