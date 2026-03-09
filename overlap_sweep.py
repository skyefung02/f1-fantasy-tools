"""
F1 Fantasy Overlap & Banking Sweep
====================================
Runs the transfer solver for every value of portfolio_max_overlap (0–8)
under both normal (2FT) and banking (1FT hard cap) scenarios, and prints
two comparison tables so you can pick the right overlap setting and assess
the cost of rolling a free transfer.

Usage
-----
    python overlap_sweep.py
    python overlap_sweep.py data.csv
    python overlap_sweep.py --pts-per-1m 1.2 --remaining-races 22
"""

import sys
import argparse

import pandas as pd
import requests

from fetch_teams import F1FantasyClient, enrich_team, load_cookie, load_settings
from solver import solve_portfolio_transfers, compute_overlap


def team_budget(team: dict) -> float:
    return round(team["total_price"] + team["budget_remaining"], 1)


def _port_combined(results: list[dict], xdelta_confidence: float) -> float:
    return sum(
        r["total_points"] + r.get("budget_value", 0.0) * xdelta_confidence
        for r in results
    )


def run_sweep(
    df: pd.DataFrame,
    current_teams: list[dict],
    budgets: list[float],
    locked: list[str],
    banned: list[str],
    budget_pts_weight: float,
    team_weights: list[float] | None,
    max_transfers: int | None = None,
    limitless_teams: list[int] | None = None,
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
                max_transfers=max_transfers,
                limitless_teams=limitless_teams,
            )
            actual_overlaps = []
            for a in range(len(results)):
                for b in range(a + 1, len(results)):
                    actual_overlaps.append(compute_overlap(results[a], results[b]))

            rows.append({
                "overlap":         overlap,
                "feasible":        True,
                "results":         results,
                "actual_overlaps": actual_overlaps,
            })
        except (RuntimeError, ValueError):
            rows.append({
                "overlap":  overlap,
                "feasible": False,
                "results":  None,
            })
    return rows


def print_sweep(
    rows: list[dict],
    budget_pts_weight: float,
    xdelta_confidence: float = 1.0,
    limitless_teams: list[int] | None = None,
) -> int | None:
    """Print overlap sweep table. Returns best_overlap."""
    limitless_set = set(limitless_teams or [])
    show_combined = budget_pts_weight != 0.0
    n_teams = len(rows[0]["results"]) if rows[0]["feasible"] else 3
    # Only show columns for non-Limitless teams — their result is constant across overlaps
    sweep_indices = [k for k in range(n_teams) if k not in limitless_set]

    sep  = "─" * (18 + len(sweep_indices) * (22 if show_combined else 12) + 14)
    wide = "=" * len(sep)

    title = "OVERLAP SWEEP  (2 free transfers)"
    if limitless_set:
        lim_labels = "+".join(f"T{k+1}" for k in sorted(limitless_set))
        title += f"  [{lim_labels} Limitless — overlap vs pre-chip picks]"
    print(f"\n{title:^{len(sep)}}")
    print(wide)

    hdr = f"  {'Overlap':>7}  "
    for k in sweep_indices:
        if show_combined:
            hdr += f"{'T'+str(k+1)+' Net':>8}  {'T'+str(k+1)+' Comb':>9}  "
        else:
            hdr += f"{'T'+str(k+1)+' Net':>8}  "
    hdr += f"{'Portfolio':>10}  {'Transfers':>10}  {'Actual OL':>10}"
    print(hdr)
    print(f"  {'─'*7}  " + "─" * (len(hdr) - 12))

    best_combined = None
    best_overlap  = None

    for row in rows:
        overlap = row["overlap"]
        if not row["feasible"]:
            print(f"  {overlap:>7}  {'— infeasible —'}")
            continue

        results       = row["results"]
        team_nets     = [r["total_points"] for r in results]
        team_combined = [r["total_points"] + r.get("budget_value", 0.0) * xdelta_confidence for r in results]
        # Portfolio score only counts non-Limitless teams (Limitless is constant)
        port_comb     = sum(team_combined[k] for k in sweep_indices)
        transfers     = "+".join(str(results[k]["n_transfers"]) for k in sweep_indices)
        actual_ol     = "/".join(str(o) for o in row["actual_overlaps"])

        if best_combined is None or port_comb > best_combined:
            best_combined = port_comb
            best_overlap  = overlap

        line = f"  {overlap:>7}  "
        for k in sweep_indices:
            if show_combined:
                line += f"{team_nets[k]:>8.1f}  {team_combined[k]:>9.1f}  "
            else:
                line += f"{team_nets[k]:>8.1f}  "
        line += f"{port_comb:>10.1f}  {transfers:>10}  {actual_ol:>10}"
        print(line)

    print(wide)

    if best_overlap is not None:
        metric = "combined (xPts + budget value)" if show_combined else "net xPts"
        print(f"\n  Highest portfolio {metric}: overlap = {best_overlap}")
        print(f"  Run:  python transfer_advisor.py --overlap {best_overlap}")

    return best_overlap


def print_banking_section(
    rows_2ft: list[dict],
    rows_1ft: list[dict],
    budget_pts_weight: float,
    xdelta_confidence: float,
    best_overlap: int | None,
    limitless_teams: list[int] | None = None,
) -> None:
    """Print banking cost table comparing 1FT vs 2FT at each overlap."""
    limitless_set = set(limitless_teams or [])
    show_combined = budget_pts_weight != 0.0
    metric        = "Combined" if show_combined else "Net xPts"

    wide = "=" * 68
    sep  = "─" * 68

    title = "FREE TRANSFER BANKING COST  (1FT hard cap vs 2FT)"
    if limitless_set:
        lim_labels = "+".join(f"T{k+1}" for k in sorted(limitless_set))
        title += f"  [excl. {lim_labels} Limitless]"
    print(f"\n{title:^68}")
    print(wide)
    print(f"  {'Overlap':>7}  {'2FT Port':>10}  {'1FT Port':>10}  {'Delta':>8}  {'Transfers(1FT)':>14}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*14}")

    def _port_sweep(results, xdc):
        """Portfolio score excluding Limitless teams (their score is constant)."""
        return sum(
            r["total_points"] + r.get("budget_value", 0.0) * xdc
            for k, r in enumerate(results)
            if k not in limitless_set
        )

    for row_2, row_1 in zip(rows_2ft, rows_1ft):
        overlap = row_2["overlap"]
        marker  = "  ◄" if overlap == best_overlap else ""

        p2_str    = "infeasible"
        p1_str    = "infeasible"
        delta_str = "       n/a"
        xfer_str  = "—"
        p2        = None

        if row_2["feasible"]:
            p2     = _port_sweep(row_2["results"], xdelta_confidence)
            p2_str = f"{p2:>10.1f}"

        if row_1["feasible"]:
            p1        = _port_sweep(row_1["results"], xdelta_confidence)
            p1_str    = f"{p1:>10.1f}"
            delta_str = f"{p1 - p2:>+8.1f}" if p2 is not None else "       n/a"
            xfer_str  = "+".join(
                str(r["n_transfers"]) for k, r in enumerate(row_1["results"])
                if k not in limitless_set
            )

        print(f"  {overlap:>7}  {p2_str}  {p1_str}  {delta_str}  {xfer_str:>14}{marker}")

    print(wide)

    # Summary at the best overlap
    if best_overlap is not None:
        row_2 = rows_2ft[best_overlap]
        row_1 = rows_1ft[best_overlap]
        if row_2["feasible"]:
            p2      = _port_sweep(row_2["results"], xdelta_confidence)
            n_sweep = sum(1 for k in range(len(row_2["results"])) if k not in limitless_set)
            print(f"\n  At optimal overlap ({best_overlap}) — banking cost:")
            print(f"    {metric} with 2FT : {p2:.1f}")
            if row_1["feasible"]:
                p1       = _port_sweep(row_1["results"], xdelta_confidence)
                delta    = p1 - p2
                per_team = delta / n_sweep if n_sweep else 0.0
                print(f"    {metric} with 1FT : {p1:.1f}  (hard cap: max 1 transfer per team)")
                print(f"    Banking cost     : {delta:+.1f} pts portfolio  ({per_team:+.1f} pts/team avg)")
            else:
                print(f"    1FT scenario     : infeasible at this overlap")
    print()


def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Sweep portfolio_max_overlap 0–8 across 2FT and 1FT scenarios."
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
    # limitless is 1-indexed in settings; convert to 0-indexed
    limitless_teams = [t - 1 for t in settings.get("limitless", [])]

    if budget_pts_weight != 0.0:
        print(f"  pts/1M/race : {args.pts_per_1m}  ×  {args.remaining_races} races  =  {budget_pts_weight:.1f} pts/M total")
        if xdelta_confidence != 1.0:
            print(f"  xdelta conf : {xdelta_confidence:.0%}  (budget value discounted in reported Combined)")
    if args.weights:
        print(f"  team weights: {args.weights}")
    if limitless_teams:
        lim_labels = "+".join(f"T{k+1}" for k in sorted(limitless_teams))
        print(f"  limitless   : {lim_labels}  (overlap enforced vs pre-chip picks; {lim_labels} excluded from sweep table)")
    print("\nRunning overlap sweep (0–8) × 2 FT scenarios (18 solves)...")

    rows_2ft = run_sweep(
        df, current_teams, budgets, locked, banned, budget_pts_weight, args.weights,
        max_transfers=None, limitless_teams=limitless_teams,
    )
    rows_1ft = run_sweep(
        df, current_teams, budgets, locked, banned, budget_pts_weight, args.weights,
        max_transfers=1, limitless_teams=limitless_teams,
    )

    best_overlap = print_sweep(rows_2ft, budget_pts_weight, xdelta_confidence=xdelta_confidence, limitless_teams=limitless_teams)
    print_banking_section(rows_2ft, rows_1ft, budget_pts_weight, xdelta_confidence, best_overlap, limitless_teams=limitless_teams)


if __name__ == "__main__":
    main()
