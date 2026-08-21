"""
wildcard_eval.py — should I play the wildcard?

For each team (or a specific team via --team), solves twice:
  1. Normal: standard transfer rules + penalty for extra transfers
  2. Wildcard: same budget, all transfers free (no penalty)

Prints a side-by-side comparison and the points gain from playing the wildcard.
"""

import argparse
import sys

import pandas as pd
import requests

from fetch_teams import F1FantasyClient, enrich_team, load_cookie, load_settings, get_current_teams
from solver import solve_portfolio_transfers
from transfer_advisor import team_budget


SEP  = "─" * 64
WIDE = "=" * 64


def _pick_set(result: dict) -> set[str]:
    return (
        {d["name"].upper() for d in result["drivers"]}
        | {c["name"].upper() for c in result["constructors"]}
    )


def _turbo(result: dict) -> str:
    for d in result["drivers"]:
        if d["is_turbo"]:
            return d["name"].upper()
    return "?"


def print_comparison(
    team_idx: int,
    current_team: dict,
    budget: float,
    normal: dict,
    wildcard: dict,
    budget_pts_weight: float,
    xdelta_confidence: float,
) -> None:
    show_delta = budget_pts_weight != 0.0

    def combined(r: dict) -> float:
        return r["total_points"] + r.get("budget_value", 0.0) * xdelta_confidence

    normal_score   = normal["total_points"]
    wildcard_score = wildcard["total_points"]
    delta_pts      = wildcard_score - normal_score

    normal_combined   = combined(normal)
    wildcard_combined = combined(wildcard)
    delta_combined    = wildcard_combined - normal_combined

    print(f"\n{WIDE}")
    print(f"  TEAM {team_idx}  —  {current_team['name']}")
    print(f"  Budget: {budget:.1f}m")
    print(WIDE)

    # ── Side-by-side picks ────────────────────────────────────────────────────
    normal_picks   = [d["name"].upper() for d in normal["drivers"]] + \
                     [c["name"].upper() for c in normal["constructors"]]
    wildcard_picks = [d["name"].upper() for d in wildcard["drivers"]] + \
                     [c["name"].upper() for c in wildcard["constructors"]]

    normal_set   = set(normal_picks)
    wildcard_set = set(wildcard_picks)
    shared       = normal_set & wildcard_set

    all_picks = sorted(set(normal_picks + wildcard_picks))

    col = 10
    print(f"\n  {'Pick':<{col}}  {'Normal':^8}  {'Wildcard':^8}")
    print(f"  {'─'*col}  {'─'*8}  {'─'*8}")
    for p in all_picks:
        n_mark = "✓" if p in normal_set   else ""
        w_mark = "✓" if p in wildcard_set else ""
        turbo_n = " ★" if p == _turbo(normal)   else ""
        turbo_w = " ★" if p == _turbo(wildcard) else ""
        print(f"  {p:<{col}}  {n_mark+turbo_n:^8}  {w_mark+turbo_w:^8}")

    # ── Transfers ─────────────────────────────────────────────────────────────
    print(f"\n  {'':30}  {'Normal':>10}  {'Wildcard':>10}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}")
    print(f"  {'Transfers':<30}  {normal['n_transfers']:>10}  {wildcard['n_transfers']:>10}")
    print(f"  {'Transfer penalty':<30}  {-normal['penalty_pts']:>+10.0f}  {-wildcard['penalty_pts']:>+10.0f}")
    print(f"  {'Gross xPts':<30}  {normal['gross_points']:>10.2f}  {wildcard['gross_points']:>10.2f}")
    print(f"  {'Net xPts':<30}  {normal_score:>10.2f}  {wildcard_score:>10.2f}")
    if show_delta:
        bv_n = normal.get("budget_value", 0.0)   * xdelta_confidence
        bv_w = wildcard.get("budget_value", 0.0) * xdelta_confidence
        print(f"  {'Budget value':<30}  {bv_n:>+10.2f}  {bv_w:>+10.2f}")
        print(f"  {'Combined score':<30}  {normal_combined:>10.2f}  {wildcard_combined:>10.2f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  Wildcard gain  :  {delta_pts:+.2f} pts net xPts")
    if show_delta:
        print(f"  Wildcard gain  :  {delta_combined:+.2f} pts combined")
    if delta_pts > 0:
        print(f"  Verdict        :  WORTH IT  (+{delta_pts:.1f} pts vs normal)")
    elif delta_pts == 0:
        print(f"  Verdict        :  NO BENEFIT  (wildcard team = normal team)")
    else:
        print(f"  Verdict        :  NOT WORTH IT  ({delta_pts:.1f} pts vs normal)")
    print(SEP)


def main() -> None:
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Compare wildcard vs normal transfers for one or all teams."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=settings.get("data_file", "sample_data.csv"),
        help="Path to xPts CSV (default from settings.json)",
    )
    parser.add_argument(
        "--team",
        type=int,
        default=None,
        metavar="N",
        help="1-indexed team number to evaluate (default: all teams)",
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
    args = parser.parse_args()

    budget_pts_weight = args.pts_per_1m * args.remaining_races
    xdelta_confidence = float(settings.get("xdelta_confidence", 1.0))
    locked = [c.upper() for c in settings.get("locked", [])]
    banned = [c.upper() for c in settings.get("banned", [])]
    # Prefer per-team values from the API; fall back to a single settings value for snapshots


    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: file '{args.csv_file}' not found.")
        sys.exit(1)

    col_aliases = {"code": "name", "xPts": "expected_points"}
    df = df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns})
    df["type"]            = df["type"].str.lower().str.strip()
    df["price"]           = pd.to_numeric(df["price"],            errors="coerce")
    df["expected_points"] = pd.to_numeric(df["expected_points"],  errors="coerce")
    if "xDeltaPrice" in df.columns:
        df["xDeltaPrice"] = pd.to_numeric(df["xDeltaPrice"], errors="coerce").fillna(0.0)

    # ── Load current teams (snapshot if available, else live API) ─────────────
    current_teams  = get_current_teams(settings)
    budgets        = [team_budget(t) for t in current_teams]
    n_teams        = len(current_teams)
    free_transfers = [t.get("free_transfers", 2) for t in current_teams]

    # Determine which teams to evaluate
    if args.team is not None:
        if args.team < 1 or args.team > n_teams:
            print(f"Error: --team must be between 1 and {n_teams}.")
            sys.exit(1)
        eval_teams = [args.team - 1]  # 0-indexed
    else:
        eval_teams = list(range(n_teams))

    print(f"\nEvaluating wildcard for: {', '.join(f'T{k+1}' for k in eval_teams)}\n")

    common_kwargs = dict(
        df=df,
        current_teams=current_teams,
        budgets=budgets,
        locked=locked,
        banned=banned,
        budget_pts_weight=budget_pts_weight,
        max_free_transfers=free_transfers,
        # Solve each team independently — no cross-team overlap constraint.
        max_pairwise_overlap=8,
    )

    summary = []  # (team_idx, delta_pts, delta_combined)

    for k in eval_teams:
        normal_results   = solve_portfolio_transfers(**common_kwargs)
        wildcard_results = solve_portfolio_transfers(**common_kwargs, wildcard_teams=[k])

        n = normal_results[k]
        w = wildcard_results[k]
        print_comparison(
            team_idx=k + 1,
            current_team=current_teams[k],
            budget=budgets[k],
            normal=n,
            wildcard=w,
            budget_pts_weight=budget_pts_weight,
            xdelta_confidence=xdelta_confidence,
        )

        def combined(r: dict) -> float:
            return r["total_points"] + r.get("budget_value", 0.0) * xdelta_confidence

        summary.append((
            k + 1,
            current_teams[k]["name"],
            w["total_points"] - n["total_points"],
            combined(w) - combined(n),
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
    show_delta = budget_pts_weight != 0.0
    print(f"\n{'─' * 64}")
    print(f"  WILDCARD SUMMARY")
    print(f"{'─' * 64}")
    if show_delta:
        print(f"  {'Team':<5}  {'Name':<24}  {'xPts gain':>10}  {'Combined gain':>14}  Verdict")
        print(f"  {'─'*5}  {'─'*24}  {'─'*10}  {'─'*14}  {'─'*20}")
    else:
        print(f"  {'Team':<5}  {'Name':<24}  {'xPts gain':>10}  Verdict")
        print(f"  {'─'*5}  {'─'*24}  {'─'*10}  {'─'*20}")
    for team_idx, name, d_pts, d_combined in summary:
        if d_pts > 0:
            verdict = f"WORTH IT  (+{d_pts:.1f} pts)"
        elif d_pts == 0:
            verdict = "NO BENEFIT"
        else:
            verdict = f"NOT WORTH IT  ({d_pts:.1f} pts)"
        if show_delta:
            print(f"  T{team_idx:<4}  {name:<24}  {d_pts:>+10.2f}  {d_combined:>+14.2f}  {verdict}")
        else:
            print(f"  T{team_idx:<4}  {name:<24}  {d_pts:>+10.2f}  {verdict}")
    print(f"{'─' * 64}\n")


if __name__ == "__main__":
    main()
