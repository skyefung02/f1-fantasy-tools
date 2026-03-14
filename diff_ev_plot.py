"""
F1 Fantasy Differential EV vs Total EV — Team Landscape
=========================================================
Enumerates many distinct feasible teams for each plot team and plots them
all on a (Total EV, Differential EV vs reference) plane, revealing the full
solution landscape.  The Pareto frontier is highlighted.

Unlike a lambda sweep, this directly explores the discrete solution space
rather than trying to parameterise it — which is more appropriate for small
ILP problems where the Pareto frontier has only a handful of distinct points.

Reference team resolution
-------------------------
  - If the reference team is NOT playing a chip, its optimal picks are
    determined by a Phase 1 solve at λ=0.
  - If the reference team IS playing a chip, supply pre-chip picks manually
    via  reference_team_picks  in settings.json.

Teams included in the plot
--------------------------
  Only teams that are (a) not the reference team and (b) not playing a chip
  are enumerated and plotted.

Usage
-----
    python diff_ev_plot.py
    python diff_ev_plot.py data.csv
    python diff_ev_plot.py --top-n 75
    python diff_ev_plot.py --pts-per-1m 1.2 --remaining-races 20
"""

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

from fetch_teams import F1FantasyClient, enrich_team, load_cookie, load_settings
from solver import solve_portfolio_transfers, solve_with_transfers, compute_differential_ev

TEAM_COLOURS = ["#E10600", "#00A0DC", "#00D2BE", "#FF8000", "#C0C0C0"]
TEAM_MARKERS = ["o", "s", "^", "D", "v"]


def team_budget(team: dict) -> float:
    return round(team["total_price"] + team["budget_remaining"], 1)


def ref_tlas_from_settings(settings: dict) -> set[str]:
    picks = settings.get("reference_team_picks", {})
    return {t.upper() for t in picks.get("drivers", []) + picks.get("constructors", [])}


def enumerate_teams(
    df: pd.DataFrame,
    current_team: dict,
    budget: float,
    locked: list[str],
    banned: list[str],
    budget_pts_weight: float,
    n: int,
) -> list[dict]:
    """
    Enumerate up to n distinct teams via repeated solve_with_transfers calls.

    Each successive solve is constrained to differ in at least 1 of the 8
    decisions (5 driver picks + 2 constructor picks + 1 turbo) from every
    previously found team, using overlap_constraints with max_overlap=7.
    Teams are returned in descending order of total_points (best first).
    """
    current_d = {d["tla"].upper() for d in current_team["drivers"]}
    current_c = {c["tla"].upper() for c in current_team["constructors"]}

    results = []
    for _ in range(n):
        try:
            result = solve_with_transfers(
                df,
                current_driver_tlas=current_d,
                current_constructor_tlas=current_c,
                budget=budget,
                locked=locked,
                banned=banned,
                overlap_constraints=[(r, 7) for r in results],
                budget_pts_weight=budget_pts_weight,
            )
            results.append(result)
        except RuntimeError:
            break
    return results


def pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Return Pareto-optimal (x, y) points where both x (total EV) and
    y (differential EV) are to be maximised.
    A point is on the frontier if no other point is >= in both dimensions.
    Returns the frontier sorted by x ascending (left to right).
    """
    if not points:
        return []
    # Sort by x descending; for equal x, by y descending
    sorted_pts = sorted(set(points), key=lambda p: (-p[0], -p[1]))
    frontier, max_y = [], float("-inf")
    for p in sorted_pts:
        if p[1] > max_y:
            frontier.append(p)
            max_y = p[1]
    return sorted(frontier, key=lambda p: p[0])


def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Plot total EV vs differential EV landscape for each plot team."
    )
    parser.add_argument(
        "csv_file", nargs="?", default=settings.get("data_file", "sample_data.csv"),
    )
    parser.add_argument(
        "--pts-per-1m", type=float,
        default=settings.get("pts_per_1m_per_race", 0.0), metavar="PTS",
    )
    parser.add_argument(
        "--remaining-races", type=int,
        default=settings.get("remaining_races", 23), metavar="N",
    )
    parser.add_argument(
        "--top-n", type=int, default=50, metavar="N",
        help="Number of distinct teams to enumerate per plot team (default 50)",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=settings.get("team_weights", None), metavar="W",
    )
    args = parser.parse_args()

    budget_pts_weight = args.pts_per_1m * args.remaining_races
    xdelta_confidence = float(settings.get("xdelta_confidence", 1.0))
    limitless_teams   = [t - 1 for t in settings.get("limitless", [])]
    limitless_set     = set(limitless_teams)
    ref_k             = settings.get("reference_team", 1) - 1  # 0-indexed

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
    n_teams       = len(current_teams)
    locked        = [c.upper() for c in settings.get("locked", [])]
    banned        = [c.upper() for c in settings.get("banned", [])]

    plot_teams = [k for k in range(n_teams) if k != ref_k and k not in limitless_set]
    if not plot_teams:
        print("No eligible teams to plot (all teams are either the reference or playing a chip).")
        sys.exit(0)

    chip_labels  = [f"T{k+1}" for k in limitless_set]
    plot_labels  = [f"T{k+1}" for k in plot_teams]
    print(f"\nReference team : T{ref_k + 1}")
    if chip_labels:
        print(f"Chip teams     : {', '.join(chip_labels)}  (excluded from plot)")
    print(f"Teams to plot  : {', '.join(plot_labels)}")

    # ── Phase 1: resolve reference picks ─────────────────────────────────────
    ref_result = None

    if ref_k in limitless_set:
        ref_tlas  = ref_tlas_from_settings(settings)
        if not ref_tlas:
            print(
                f"\nError: T{ref_k+1} is playing a chip — supply pre-chip picks in "
                "settings.json under 'reference_team_picks'."
            )
            sys.exit(1)
        ref_label = f"T{ref_k + 1} (pre-chip, manual)"
        print(f"Reference picks: {', '.join(sorted(ref_tlas))}  [manual override]")
    else:
        print(f"\nPhase 1: solving T{ref_k + 1} at λ=0 to establish reference picks...")
        baseline   = solve_portfolio_transfers(
            df,
            current_teams=current_teams,
            budgets=budgets,
            locked=locked,
            banned=banned,
            budget_pts_weight=budget_pts_weight,
            team_weights=args.weights,
            limitless_teams=limitless_teams,
            diff_ev_weight=0.0,
        )
        ref_result = baseline[ref_k]
        ref_tlas   = (
            {d["name"].upper() for d in ref_result["drivers"]}
            | {c["name"].upper() for c in ref_result["constructors"]}
        )
        ref_label  = f"T{ref_k + 1} (optimal, λ=0)"
        print(f"Reference picks: {', '.join(sorted(ref_tlas))}")

    # ── Phase 2: enumerate teams for each plot team ───────────────────────────
    use_combined = budget_pts_weight != 0.0
    x_key        = "combined" if use_combined else "total_ev"

    # team_points[k] = list of (x, y, result_dict)
    team_points: dict[int, list[tuple[float, float, dict]]] = {}

    for k in plot_teams:
        print(f"\nEnumerating up to {args.top_n} teams for T{k + 1}...")
        enumerated = enumerate_teams(
            df,
            current_team=current_teams[k],
            budget=budgets[k],
            locked=locked,
            banned=banned,
            budget_pts_weight=budget_pts_weight,
            n=args.top_n,
        )
        print(f"  Found {len(enumerated)} distinct teams.")

        points = []
        for result in enumerated:
            diff_ev  = compute_differential_ev(result, ref_tlas)
            net_pts  = result["total_points"]
            combined = net_pts + result.get("budget_value", 0.0) * xdelta_confidence
            x        = combined if use_combined else net_pts
            points.append((x, diff_ev, result))
        team_points[k] = points

    # ── Print Pareto frontier summary ─────────────────────────────────────────
    for k in plot_teams:
        pts      = team_points[k]
        frontier = pareto_frontier([(x, y) for x, y, _ in pts])
        print(f"\nPareto frontier for T{k + 1}  ({len(frontier)} points):")
        x_label_short = "Combined" if use_combined else "Total EV"
        print(f"  {'#':<4}  {x_label_short:>10}  {'Diff EV':>8}  {'Ratio':>7}  {'Transfers':>10}  Assets")
        print(f"  {'─'*4}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*30}")
        frontier_set = set(frontier)
        cur_d = {d["tla"].upper() for d in current_teams[k]["drivers"]}
        cur_c = {c["tla"].upper() for c in current_teams[k]["constructors"]}
        for i, (x, y) in enumerate(sorted(frontier, key=lambda p: -p[0]), 1):
            # Find matching result
            match = next(r for rx, ry, r in pts if abs(rx - x) < 0.01 and abs(ry - y) < 0.01)
            ratio     = y / match["total_points"] * 100 if match["total_points"] > 0 else 0
            drivers   = [d["name"] for d in match["drivers"]]
            constrs   = [c["name"] for c in match["constructors"]]
            turbo     = match["turbo_driver"]
            xfers     = match["n_transfers"]
            assets    = ", ".join(drivers + constrs) + f"  [turbo: {turbo}]"
            new_d = {d["name"].upper() for d in match["drivers"]}
            new_c = {c["name"].upper() for c in match["constructors"]}
            ins  = sorted((new_d - cur_d) | (new_c - cur_c))
            outs = sorted((cur_d - new_d) | (cur_c - new_c))
            xfer_str = f"IN: {', '.join(ins) or '—'}  OUT: {', '.join(outs) or '—'}"
            print(f"  {i:<4}  {x:>10.1f}  {y:>8.2f}  {ratio:>6.1f}%  {xfers:>10}  {assets}")
            print(f"  {'':4}  {'':10}  {'':8}  {'':7}  {'':10}  {xfer_str}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.tick_params(colors="#cccccc")
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.title.set_color("#ffffff")

    for k in plot_teams:
        pts    = team_points[k]
        colour = TEAM_COLOURS[k % len(TEAM_COLOURS)]
        marker = TEAM_MARKERS[k % len(TEAM_MARKERS)]

        all_x = [p[0] for p in pts]
        all_y = [p[1] for p in pts]
        frontier = pareto_frontier(list(zip(all_x, all_y)))
        frontier_set = set(frontier)

        # Full cloud — faded
        ax.scatter(
            all_x, all_y,
            color=colour, s=25, marker=marker, alpha=0.25, zorder=3,
        )

        # Pareto frontier — bright, connected
        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.plot(fx, fy, color=colour, linewidth=1.5, alpha=0.8, zorder=4)
            ax.scatter(
                fx, fy,
                color=colour, s=80, marker=marker, zorder=5,
                label=f"T{k + 1} frontier",
            )
            # Label the extreme frontier points with ratio
            x_axis_label_short = "Combined EV" if use_combined else "Total EV"
            extreme_points = [
                (frontier[0],  "Max Diff EV",              ( 0,  12)),
                (frontier[-1], f"Max {x_axis_label_short}", ( 0, -28)),
            ]
            for (fpx, fpy), base_lbl, (xoff, yoff) in extreme_points:
                match = next(r for rx, ry, r in pts if abs(rx - fpx) < 0.01 and abs(ry - fpy) < 0.01)
                ratio = fpy / match["total_points"] * 100 if match["total_points"] > 0 else 0.0
                lbl = f"{base_lbl} — {ratio:.1f}%"
                ax.annotate(
                    lbl, (fpx, fpy),
                    textcoords="offset points", xytext=(xoff, yoff),
                    fontsize=7.5, color=colour, ha="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec=colour, lw=0.6, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.6),
                )

    # Reference team anchor (diff EV vs itself = 0)
    ref_colour = TEAM_COLOURS[ref_k % len(TEAM_COLOURS)]
    if ref_result is not None:
        ref_x = ref_result["total_points"] + ref_result.get("budget_value", 0.0) * xdelta_confidence
    else:
        ref_x = sum(
            float(df.loc[df["name"].str.upper() == tla, "expected_points"].iloc[0])
            for tla in ref_tlas
            if not df.loc[df["name"].str.upper() == tla, "expected_points"].empty
        )
    ax.scatter(
        [ref_x], [0.0],
        color=ref_colour, s=200, marker="*", zorder=6, label=ref_label,
    )
    ax.annotate(
        ref_label, (ref_x, 0.0),
        textcoords="offset points", xytext=(8, 6),
        fontsize=8.5, color=ref_colour,
    )

    x_axis_label = (
        f"Combined EV  (net xPts + budget value × {xdelta_confidence:.0%})"
        if use_combined
        else "Total EV  (net xPts after transfer penalties)"
    )
    ax.set_xlabel(x_axis_label, fontsize=11)
    ax.set_ylabel(f"Differential EV  (vs T{ref_k + 1})", fontsize=11)
    ax.set_title(
        "Total EV vs Differential EV — Team Landscape",
        fontsize=13, fontweight="bold", color="#ffffff",
    )
    ax.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="#cccccc", fontsize=10)
    ax.grid(True, color="#333333", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.text(
        0.01, 0.01,
        f"Reference: {ref_label}  —  {', '.join(sorted(ref_tlas))}",
        transform=ax.transAxes, fontsize=7.5, color="#777777",
        verticalalignment="bottom",
    )
    ax.text(
        0.99, 0.99,
        "Faded points = all enumerated teams   Bright line = Pareto frontier",
        transform=ax.transAxes, fontsize=7.5, color="#666666",
        verticalalignment="top", horizontalalignment="right",
    )

    plt.tight_layout()

    save_path = Path(__file__).parent / "tmp" / "diff_ev_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
