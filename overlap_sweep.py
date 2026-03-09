"""
F1 Fantasy Overlap Sweep
========================
Runs the transfer solver for every value of portfolio_max_overlap (0–8) and
prints a compact comparison table, so you can pick the right overlap setting
before running transfer_advisor.py for full transfer details.

Usage
-----
    python overlap_sweep.py
    python overlap_sweep.py data.csv
    python overlap_sweep.py --pts-per-1m 1.2 --remaining-races 22
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import requests

from fetch_teams import F1FantasyClient, enrich_team, load_cookie, load_settings
from solver import solve_portfolio_transfers, compute_overlap


def team_budget(team: dict) -> float:
    return round(team["total_price"] + team["budget_remaining"], 1)


def run_sweep(
    df: pd.DataFrame,
    current_teams: list[dict],
    budgets: list[float],
    locked: list[str],
    banned: list[str],
    budget_pts_weight: float,
    team_weights: list[float] | None,
) -> list[dict]:
    """Run the solver for overlap values 0..8, return list of result rows."""
    rows = []
    for overlap in range(9):
        try:
            results = solve_portfolio_transfers(
                df,
                current_teams=current_teams,
                budgets=budgets,
                max_pairwise_overlap=overlap,
                locked=locked,
                banned=banned,
                budget_pts_weight=budget_pts_weight,
                team_weights=team_weights,
            )
            actual_overlaps = []
            for a in range(len(results)):
                for b in range(a + 1, len(results)):
                    actual_overlaps.append(compute_overlap(results[a], results[b]))

            rows.append({
                "overlap":          overlap,
                "feasible":         True,
                "results":          results,
                "actual_overlaps":  actual_overlaps,
            })
        except (RuntimeError, ValueError):
            rows.append({
                "overlap":  overlap,
                "feasible": False,
                "results":  None,
            })
    return rows


def print_sweep(rows: list[dict], budget_pts_weight: float, xdelta_confidence: float = 1.0) -> None:
    show_combined = budget_pts_weight != 0.0
    n_teams = len(rows[0]["results"]) if rows[0]["feasible"] else 3

    sep  = "─" * (18 + n_teams * (22 if show_combined else 12) + 14)
    wide = "=" * len(sep)

    print(f"\n{'OVERLAP SWEEP':^{len(sep)}}")
    print(wide)

    # Header
    hdr = f"  {'Overlap':>7}  "
    for k in range(n_teams):
        if show_combined:
            hdr += f"{'T'+str(k+1)+' Net':>8}  {'T'+str(k+1)+' Comb':>9}  "
        else:
            hdr += f"{'T'+str(k+1)+' Net':>8}  "
    hdr += f"{'Portfolio':>10}  {'Transfers':>10}  {'Actual OL':>10}"
    print(hdr)
    print(f"  {'─'*7}  " + ("─"*(len(hdr)-12)))

    best_combined = None
    best_overlap  = None

    for row in rows:
        overlap = row["overlap"]
        if not row["feasible"]:
            line = f"  {overlap:>7}  {'— infeasible —'}"
            print(line)
            continue

        results  = row["results"]
        team_nets     = [r["total_points"] for r in results]
        team_combined = [r["total_points"] + r.get("budget_value", 0.0) * xdelta_confidence for r in results]
        portfolio_combined = sum(team_combined)
        transfers = "+".join(str(r["n_transfers"]) for r in results)
        actual_ol = "/".join(str(o) for o in row["actual_overlaps"])

        if best_combined is None or portfolio_combined > best_combined:
            best_combined = portfolio_combined
            best_overlap  = overlap

        line = f"  {overlap:>7}  "
        for k in range(n_teams):
            if show_combined:
                line += f"{team_nets[k]:>8.1f}  {team_combined[k]:>9.1f}  "
            else:
                line += f"{team_nets[k]:>8.1f}  "
        line += f"{portfolio_combined:>10.1f}  {transfers:>10}  {actual_ol:>10}"
        print(line)

    print(wide)

    if best_overlap is not None:
        metric = "combined (xPts + budget value)" if show_combined else "net xPts"
        print(f"\n  Highest portfolio {metric}: overlap = {best_overlap}")
        print(f"  Run:  python transfer_advisor.py --overlap {best_overlap}\n")


def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Sweep portfolio_max_overlap 0–8 and compare solver outcomes."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=settings.get("data_file", "sample_data.csv"),
    )
    parser.add_argument(
        "--pts-per-1m",
        type=float,
        default=settings.get("pts_per_1m_per_race", 0.0),
        metavar="PTS",
    )
    parser.add_argument(
        "--remaining-races",
        type=int,
        default=settings.get("remaining_races", 23),
        metavar="N",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=settings.get("team_weights", None),
        metavar="W",
    )
    args = parser.parse_args()

    budget_pts_weight = args.pts_per_1m * args.remaining_races

    # ── Load xPts CSV ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: file '{args.csv_file}' not found.")
        sys.exit(1)

    col_aliases = {"code": "name", "xPts": "expected_points"}
    df = df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns})

    required = {"name", "type", "price", "expected_points"}
    missing  = required - set(df.columns)
    if missing:
        print(f"Error: CSV missing columns: {missing}")
        sys.exit(1)

    df["type"]            = df["type"].str.lower().str.strip()
    df["price"]           = pd.to_numeric(df["price"],            errors="coerce")
    df["expected_points"] = pd.to_numeric(df["expected_points"],  errors="coerce")
    if "xDeltaPrice" in df.columns:
        df["xDeltaPrice"] = pd.to_numeric(df["xDeltaPrice"], errors="coerce").fillna(0.0)

    # ── Fetch current teams ───────────────────────────────────────────────────
    cookie = load_cookie()
    if not cookie:
        print("Error: F1_FANTASY_COOKIE not set.")
        sys.exit(1)

    user_uuid = settings.get("user_uuid")
    if not user_uuid:
        print("Error: 'user_uuid' not set in settings.json.")
        sys.exit(1)

    gameday = settings.get("gameday", 1)
    client  = F1FantasyClient(cookie=cookie)

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
        sys.exit(1)

    if not raw_teams:
        print("No teams found.")
        sys.exit(0)

    current_teams = [enrich_team(raw, players) for raw in raw_teams]
    budgets       = [team_budget(t) for t in current_teams]
    locked        = [c.upper() for c in settings.get("locked", [])]
    banned        = [c.upper() for c in settings.get("banned", [])]

    xdelta_confidence = float(settings.get("xdelta_confidence", 1.0))

    if budget_pts_weight != 0.0:
        print(f"  pts/1M/race : {args.pts_per_1m}  ×  {args.remaining_races} races  =  {budget_pts_weight:.1f} pts/M total")
        if xdelta_confidence != 1.0:
            print(f"  xdelta conf : {xdelta_confidence:.0%}  (budget value discounted in reported Combined)")
    if args.weights:
        print(f"  team weights: {args.weights}")
    print("\nRunning overlap sweep (0–8)...")

    rows = run_sweep(
        df, current_teams, budgets, locked, banned, budget_pts_weight, args.weights
    )
    print_sweep(rows, budget_pts_weight, xdelta_confidence=xdelta_confidence)


if __name__ == "__main__":
    main()
