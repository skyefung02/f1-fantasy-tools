"""
F1 Fantasy Week Planner — EV vs Differentiation Frontier
==========================================================
Traces the full frontier of 3-team portfolios trading total EV against
differentiation, in one joint solve per point:

    maximise   portfolio EV
    subject to total shared EV (summed over every team pair)  ≤  cap

The cap is swept downwards from the unconstrained optimum until the problem
becomes infeasible. Shared EV between two teams = xPts of every shared driver
and constructor, plus the turbo driver's xPts again if both turbo the same
driver — the xPts-weighted version of portfolio_max_overlap.

Each frontier row is a complete portfolio (all 3 teams with transfers):

    Port EV    net xPts + budget value, summed over the teams
    Cost       Port EV given up vs the unconstrained optimum (row 1), split into
               Pts now (points this week) and Budget (budget value)
    Diff gain  total shared EV removed vs row 1 — differentiation bought;
               Cut is the same as a share of row 1's total shared EV
    vs row above: EV lost / diff gained
               what moving one row down costs and buys
    Turbos     distinct turbo drivers across the portfolio
    Hits       transfer penalty points taken
    Changes    lineup swaps vs row 1's portfolio (* = turbo)

Every row gains differentiation over the row above (dominated rows are dropped).
"Chosen if r" gives the exchange rates r — EV you'd give up per point of diff
gain — under which that row maximises EV + r × diff gain. Rows without a ▸ sit
in a dent of the frontier: no single rate picks them, but they are real options.

Usage
-----
    python plan_week.py                  # solve and print the frontier
    python plan_week.py --rate 0.4       # also highlight the row a rate of 0.4 picks
    python plan_week.py --pick 3         # print full transfer plans for row 3 (no re-solve)
    python plan_week.py --pick 3 --log   # ...and log the choice + implied rate

Each table run also writes plan_week_frontier.html — the same table, to keep open
beside the terminal while --pick prints transfer plans.

--pick reads the frontier saved by the last table run, so it is instant. It
refuses if the projections, teams, settings or solve flags have changed since
that run — re-run the table first so the row numbers mean what you saw.
    python plan_week.py --step 5         # finer sweep (more solves)
    python plan_week.py --max-transfers 1   # banking scenario (1FT hard cap)
"""

import csv
import sys
import json
import hashlib
import argparse
from html import escape
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_teams import load_settings, get_current_teams
from solver import solve_portfolio_transfers, compute_overlap
from transfer_advisor import print_transfer_plan, print_portfolio_summary

LOG_PATH   = Path(__file__).parent / "plan_week_log.csv"
CACHE_PATH = Path(__file__).parent / "tmp" / "plan_week_frontier.json"
HTML_PATH  = Path(__file__).parent / "plan_week_frontier.html"


def team_budget(team: dict) -> float:
    return round(team["total_price"] + team["budget_remaining"], 1)


def load_projections(csv_file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: file '{csv_file}' not found.")
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
    return df


# ── Shared EV ─────────────────────────────────────────────────────────────────

def _picks_from_result(result: dict) -> tuple[set[str], set[str], str]:
    return (
        {d["name"].upper() for d in result["drivers"]},
        {c["name"].upper() for c in result["constructors"]},
        (result["turbo_driver"] or "").upper(),
    )


def _picks_from_current(team: dict) -> tuple[set[str], set[str], str]:
    turbo = next((d["tla"] for d in team["drivers"] if d["is_turbo"]), "")
    return (
        {d["tla"].upper() for d in team["drivers"]},
        {c["tla"].upper() for c in team["constructors"]},
        turbo.upper(),
    )


def pairwise_shared_ev(
    results: list[dict],
    current_teams: list[dict],
    limitless_set: set[int],
    xpts: dict[str, float],
) -> dict[tuple[int, int], float]:
    """
    Shared EV for every team pair, measured the same way the solver constrains it.
    Limitless teams are measured on their pre-chip picks; Limitless–Limitless
    pairs are skipped (both revert, nothing persists).
    """
    picks = [
        _picks_from_current(current_teams[k]) if k in limitless_set else _picks_from_result(r)
        for k, r in enumerate(results)
    ]
    shared = {}
    for a in range(len(results)):
        for b in range(a + 1, len(results)):
            if a in limitless_set and b in limitless_set:
                continue
            d_a, c_a, t_a = picks[a]
            d_b, c_b, t_b = picks[b]
            ev = sum(xpts.get(x, 0.0) for x in (d_a & d_b) | (c_a & c_b))
            if t_a and t_a == t_b:
                ev += xpts.get(t_a, 0.0)
            shared[(a, b)] = round(ev, 2)
    return shared


# ── Frontier ──────────────────────────────────────────────────────────────────

def run_frontier(
    df: pd.DataFrame,
    current_teams: list[dict],
    solve_kwargs: dict,
    limitless_set: set[int],
    budget_pts_weight: float,
    xdelta_confidence: float,
    step: float,
    max_solves: int,
) -> list[dict]:
    """Sweep the total shared-EV cap downward, one joint solve per step."""
    xpts = dict(zip(df["name"].str.upper(), df["expected_points"]))
    rows = []
    cap  = None

    for n in range(max_solves):
        cap_label = "none" if cap is None else f"{cap:.1f}"
        print(f"  solve {n + 1:>2}: cap {cap_label:>6} ...", end="", flush=True)
        try:
            results = solve_portfolio_transfers(
                df,
                current_teams=current_teams,
                max_pairwise_overlap=8,
                max_total_shared_ev=cap,
                **solve_kwargs,
            )
        except (RuntimeError, ValueError):
            print(" infeasible — frontier complete")
            break

        shared = pairwise_shared_ev(results, current_teams, limitless_set, xpts)
        if not shared:
            print(" no constrained pairs — nothing to sweep")
            break

        total_shared = round(sum(shared.values()), 2)
        port_ev = sum(
            r["total_points"] + (r.get("budget_value", 0.0) * xdelta_confidence if budget_pts_weight else 0.0)
            for k, r in enumerate(results)
            if k not in limitless_set
        )
        print(f" EV {port_ev:.1f}, total shared {total_shared:.1f}")
        rows.append({
            "results":      results,
            "shared":       shared,
            "total_shared": total_shared,
            "worst_pair":   max(shared.values()),
            "port_ev":      round(port_ev, 2),
        })
        cap = total_shared - step
    else:
        print(f"  stopped at --max-solves {max_solves}; frontier may continue below cap {cap:.1f}")

    return rows


def pareto_filter(rows: list[dict]) -> list[dict]:
    """
    Keep rows no other row beats on both axes (higher EV, lower total shared),
    then attach cost and diff gain relative to the top (max-EV) row.
    Returned least-differentiated first (highest EV first).
    """
    frontier, best_ev = [], float("-inf")
    for row in sorted(rows, key=lambda r: (r["total_shared"], -r["port_ev"])):
        if row["port_ev"] > best_ev + 1e-6:
            frontier.append(row)
            best_ev = row["port_ev"]
    frontier.reverse()

    top = frontier[0]
    for row in frontier:
        row["cost"]      = round(top["port_ev"] - row["port_ev"], 2)
        row["diff_gain"] = round(top["total_shared"] - row["total_shared"], 2)
    return frontier


def add_rate_ranges(frontier: list[dict]) -> None:
    """
    Row i maximises EV + r × diff_gain for r in [r_lo, r_hi]:
      r_lo = steepest cost-per-gain to reach it from any less-differentiated row
             (you must value differentiation at least this much to move here)
      r_hi = shallowest cost-per-gain from it to any more-differentiated row
    r_lo > r_hi means no single rate picks the row.
    """
    for row in frontier:
        c_i, g_i = row["cost"], row["diff_gain"]
        lo = max([(c_i - o["cost"]) / (g_i - o["diff_gain"]) for o in frontier if o["diff_gain"] < g_i] + [0.0])
        hi = min(
            [(o["cost"] - c_i) / (o["diff_gain"] - g_i) for o in frontier if o["diff_gain"] > g_i],
            default=float("inf"),
        )
        row["r_lo"], row["r_hi"] = lo, hi
        row["supported"] = lo <= hi + 1e-9


def pick_for_rate(frontier: list[dict], rate: float) -> int:
    return max(range(len(frontier)), key=lambda i: rate * frontier[i]["diff_gain"] - frontier[i]["cost"])


# ── Frontier cache ────────────────────────────────────────────────────────────

def input_fingerprint(csv_file: str, current_teams: list[dict], solve_kwargs: dict, extra: dict) -> str:
    """Hash of everything that determines the frontier — a changed hash means stale rows."""
    payload = json.dumps(
        {
            "csv":    hashlib.sha256(Path(csv_file).read_bytes()).hexdigest(),
            "teams":  current_teams,
            "solve":  solve_kwargs,
            "extra":  extra,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _json_default(o):
    # numpy scalars from the projections DataFrame
    return o.item() if hasattr(o, "item") else str(o)


def save_frontier(frontier: list[dict], fingerprint: str) -> None:
    rows = [
        {**row, "shared": [[a, b, v] for (a, b), v in row["shared"].items()]}
        for row in frontier
    ]
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(
        {"fingerprint": fingerprint, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frontier": rows},
        default=_json_default,
    ))


def load_frontier(fingerprint: str) -> list[dict]:
    if not CACHE_PATH.exists():
        print("\nError: no saved frontier. Run 'python plan_week.py' (with the same flags) first.")
        sys.exit(1)
    data = json.loads(CACHE_PATH.read_text())
    if data.get("fingerprint") != fingerprint:
        print(
            "\nError: the saved frontier is stale — projections, teams, settings or flags have\n"
            "changed since it was solved. Re-run 'python plan_week.py' with the same flags,\n"
            "check the table, then pick again."
        )
        sys.exit(1)
    frontier = data["frontier"]
    for row in frontier:
        row["shared"] = {(a, b): v for a, b, v in row["shared"]}
        for key in ("r_lo", "r_hi"):
            row[key] = float(row[key])
    print(f"Using frontier solved at {data['saved_at']}  ({CACHE_PATH.name})")
    return frontier


# ── Display ───────────────────────────────────────────────────────────────────

def _fmt_rate(r: float) -> str:
    return f"{r:.2f}" if r >= 0.1 else f"{r:.3f}"


def _fmt_rate_range(row: dict) -> str:
    if not row["supported"]:
        return "never"
    if row["r_hi"] == float("inf"):
        return f"> {_fmt_rate(row['r_lo'])}"
    if row["r_lo"] == 0.0:
        return f"< {_fmt_rate(row['r_hi'])}"
    return f"{_fmt_rate(row['r_lo'])} – {_fmt_rate(row['r_hi'])}"


def _team_changes(base: dict, new: dict) -> list[str]:
    """Lineup swaps turning one team's base result into new, priciest first; * marks the turbo."""
    def tag(asset: dict) -> str:
        return asset["name"] + ("*" if asset.get("is_turbo") else "")

    swaps = []
    for key in ("drivers", "constructors"):
        old = {a["name"]: a for a in base[key]}
        cur = {a["name"]: a for a in new[key]}
        outs = sorted((old[n] for n in old.keys() - cur.keys()), key=lambda a: -a["price"])
        ins  = sorted((cur[n] for n in cur.keys() - old.keys()), key=lambda a: -a["price"])
        swaps += [f"{tag(o)}→{tag(n)}" for o, n in zip(outs, ins)]

    # Turbo moved onto a driver both lineups hold — no swap above shows it
    turbo = new["turbo_driver"]
    if turbo != base["turbo_driver"] and any(d["name"] == turbo for d in base["drivers"]):
        swaps.append(f"turbo→{turbo}")
    return swaps


def frontier_table(frontier: list[dict], n_teams: int, limitless_set: set[int]) -> dict:
    """Display values behind the frontier table — shared by the terminal and HTML views."""
    sweep_teams = [k for k in range(n_teams) if k not in limitless_set]
    top = frontier[0]

    def pts_now(row: dict) -> float:
        return sum(row["results"][k]["total_points"] for k in sweep_teams)

    top_pts    = pts_now(top)
    top_budget = top["port_ev"] - top_pts

    rows = []
    seen_changes: dict[tuple, int] = {}
    prev = None
    for i, row in enumerate(frontier, 1):
        changes = []
        for k in sweep_teams:
            swaps = _team_changes(top["results"][k], row["results"][k])
            if not swaps:
                continue
            # Long changes already spelled out on an earlier row are referenced, not repeated
            first = seen_changes.setdefault((k, *swaps), i)
            text  = f"as row {first}" if first != i and len(swaps) > 2 else ", ".join(swaps)
            changes.append((k + 1, text))

        rows.append({
            "n":       i,
            "row":     row,
            "pts":     None if prev is None else pts_now(row) - top_pts,
            "budget":  None if prev is None else (row["port_ev"] - pts_now(row)) - top_budget,
            "lost":    None if prev is None else row["cost"] - prev["cost"],
            "gain":    None if prev is None else row["diff_gain"] - prev["diff_gain"],
            "cut":     row["diff_gain"] / top["total_shared"] if top["total_shared"] else 0.0,
            "turbos":  len({row["results"][k]["turbo_driver"] for k in sweep_teams}),
            "hits":    sum(r["penalty_pts"] for r in row["results"]),
            "changes": changes,
        })
        prev = row

    return {"top_pts": top_pts, "top_budget": top_budget, "total_shared": top["total_shared"], "rows": rows}


def _legend(metric: str) -> list[tuple[str, list[str]]]:
    return [
        ("Port EV",      [f"{metric}, summed over non-Limitless teams"]),
        ("Cost",         ["Port EV given up vs row 1, split into points this week (Pts now)",
                          "and budget value (Budget):  Pts now + Budget = −Cost"]),
        ("Diff gain",    ["total shared EV (all pairs) removed vs row 1 — differentiation bought;",
                          "Cut is the same as a share of row 1's total shared EV"]),
        ("vs row above", ["EV lost and diff gained moving down one row"]),
        ("Turbos",       ["distinct turbo drivers across the portfolio"]),
        ("Hits",         ["transfer penalty points taken, all teams"]),
        ("Chosen if r",  ["rates r (EV per diff point) under which this row maximises",
                          "EV + r × diff gain. ▸ rows are picked by some rate; · rows sit in a",
                          "dent of the frontier — no rate picks them, but they are real options"]),
        ("Changes",      ["lineup swaps vs row 1's portfolio, * = turbo"]),
    ]


def _fmt_opt(value: float | None, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def print_frontier(table: dict, rate_pick: int | None, metric: str) -> None:
    rate_w = 14 if rate_pick is None else 22   # room for the ◄ rate marker
    group = f"{'':17}{' vs row 1 ':─^42}  {' vs row above ':─^17}"
    hdr = (
        f"  {'#':>3}  {'Port EV':>8}  {'Cost':>6}  {'Pts now':>8}  {'Budget':>7}  "
        f"{'Diff gain':>9}  {'Cut':>4}  {'EV lost':>8}  {'Diff +':>7}  "
        f"{'Turbos':>6}  {'Hits':>5}  {'Chosen if r':<{rate_w}}  Changes vs row 1"
    )
    width = len(hdr)

    print(f"\n{'EV vs DIFFERENTIATION FRONTIER':^{width}}")
    print("=" * width)
    print(
        f"  Baseline (row 1): {table['top_pts']:.1f} pts + {table['top_budget']:.1f} budget value"
        f"   |   total shared EV {table['total_shared']:.1f}\n"
    )
    print(group)
    print(hdr)
    print("  " + "─" * (width - 2))

    for t in table["rows"]:
        row     = t["row"]
        rate    = _fmt_rate_range(row) if row["supported"] else "·"
        marker  = "  ◄ rate" if rate_pick == t["n"] - 1 else ""
        changes = " | ".join(f"T{team}: {text}" for team, text in t["changes"])
        hits    = f"-{t['hits']:g}" if t["hits"] else "—"
        print(
            f" {'▸' if row['supported'] else ' '}{t['n']:>3}  {row['port_ev']:>8.1f}  {row['cost']:>6.1f}  "
            f"{_fmt_opt(t['pts'], '+.1f'):>8}  {_fmt_opt(t['budget'], '+.1f'):>7}  "
            f"{row['diff_gain']:>9.1f}  {t['cut']:>4.0%}  "
            f"{_fmt_opt(t['lost'], '.1f'):>8}  {_fmt_opt(t['gain'], '.1f'):>7}  "
            f"{t['turbos']:>6}  {hits:>5}  {rate + marker:<{rate_w}}  "
            f"{changes or '—'}"
        )

    print("=" * width)
    for term, text in _legend(metric):
        print(f"  {term:<12}: {text[0]}")
        for extra in text[1:]:
            print(f"  {'':<12}  {extra}")
    print("\n  Next: python plan_week.py --pick N   (add --log to record the implied rate)")


_HTML_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --line: #d0d7de; --head: #f6f8fa;
  --accent: #0969da; --tint: rgba(9, 105, 218, .08); --bar: rgba(9, 105, 218, .16);
  --pick: rgba(191, 135, 0, .18); --pos: #1a7f37; --neg: #cf222e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8d96a0; --line: #30363d; --head: #161b22;
    --accent: #4493f8; --tint: rgba(68, 147, 248, .12); --bar: rgba(68, 147, 248, .25);
    --pick: rgba(210, 153, 34, .22); --pos: #3fb950; --neg: #f85149;
  }
}
body { margin: 0; padding: 16px 20px 32px; background: var(--bg); color: var(--fg);
       font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 16px; margin: 0 0 2px; }
.meta, .baseline { color: var(--muted); margin: 0 0 4px; }
.baseline b { color: var(--fg); font-weight: 600; }
table { border-collapse: separate; border-spacing: 0; font-variant-numeric: tabular-nums; }
table.frontier { margin-top: 16px; }
thead { position: sticky; top: 0; z-index: 1; }
th { background: var(--head); font-weight: 600; text-align: right; white-space: nowrap;
     padding: 4px 10px; border-bottom: 1px solid var(--line); }
th.group { text-align: center; color: var(--muted); font-weight: 500; }
th.left, td.left { text-align: left; }
td { padding: 3px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }
.edge { border-left: 1px solid var(--line); }
tbody tr:hover td { background-color: var(--tint); }
tr.hull td { background-color: var(--tint); }
tr.hull td.n { color: var(--accent); font-weight: 700; }
tr.pick td { background-color: var(--pick); }
td.n { color: var(--muted); }
.pos { color: var(--pos); } .neg { color: var(--neg); } .dim { color: var(--muted); }
td.cut { background-image: linear-gradient(to right, var(--bar) var(--w), transparent var(--w)); }
td.changes { white-space: normal; min-width: 260px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.team { display: inline-block; margin-right: 12px; }
.team b { color: var(--accent); font-family: system-ui, sans-serif; margin-right: 4px; }
.teams { display: flex; gap: 14px; margin-top: 12px; }
.card { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.card h2 { margin: 0; padding: 5px 10px; font-size: 13px; font-weight: 600; background: var(--head); }
.card h2 b { color: var(--accent); margin-right: 6px; }
.card th { font-weight: 500; color: var(--muted); }
.card td.asset { font-weight: 600; min-width: 120px; }
.card tr.total td { border-bottom: 0; font-weight: 600; }
.tag { margin-left: 6px; padding: 0 6px; border-radius: 8px; background: var(--accent); color: var(--bg); font-size: 11px; }
.tag.new { background: transparent; color: var(--muted); box-shadow: inset 0 0 0 1px var(--line); font-weight: 400; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; margin: 18px 0 0; color: var(--muted); max-width: 900px; }
dt { font-weight: 600; color: var(--fg); } dd { margin: 0; }
"""


def _baseline_html(results: list[dict], team_names: list[str]) -> str:
    """Row 1's portfolio as one card per team, side by side — what "Changes vs row 1" is measured against."""
    cards = []
    for k, (result, name) in enumerate(zip(results, team_names), start=1):
        new  = set(result["driver_transfers_in"]) | set(result["constructor_transfers_in"])
        chip = " ★ Limitless" if result.get("is_limitless") else " ★ Wildcard" if result.get("is_wildcard") else ""
        rows = []
        for asset in result["drivers"] + result["constructors"]:
            turbo = asset.get("is_turbo", False)
            delta = asset["xDeltaPrice"]
            css   = "pos" if delta >= 0.005 else "neg" if delta <= -0.005 else "dim"
            rows.append(
                f'<tr><td class="left asset">{escape(asset["name"])}'
                + ('<span class="tag">turbo</span>' if turbo else "")
                + ('<span class="tag new">new</span>' if asset["name"].upper() in new else "")
                + f'</td><td>{asset["price"]:.1f}</td>'
                f'<td>{asset["expected_points"] * (2 if turbo else 1):.1f}</td>'
                f'<td class="{css}">{delta:+.2f}</td></tr>'
            )
        delta_total = sum(a["xDeltaPrice"] for a in result["drivers"] + result["constructors"])
        hit = f' <span class="neg">(−{result["penalty_pts"]:g} hit)</span>' if result["penalty_pts"] else ""
        cards.append(
            f'<div class="card"><h2><b>T{k}</b>{escape(name)}{chip}</h2><table>'
            '<tr><th class="left">Asset</th><th>Price</th><th>xPts</th><th>ΔPrice</th></tr>'
            + "".join(rows)
            + f'<tr class="total"><td class="left">Net{hit}</td><td>{result["total_price"]:.1f}</td>'
            f'<td>{result["total_points"]:.1f}</td><td>{delta_total:+.2f}</td></tr></table></div>'
        )
    return f'<div class="teams">{"".join(cards)}</div>'


def write_frontier_html(
    table: dict, rate_pick: int | None, metric: str, meta: list[str], team_names: list[str],
) -> None:
    """Write the frontier table as a standalone page — it stays open while --pick scrolls the terminal."""
    def signed(value: float | None) -> str:
        if value is None:
            return '<td class="dim">—</td>'
        css = "pos" if value >= 0.05 else "neg" if value <= -0.05 else "dim"
        return f'<td class="{css}">{value:+.1f}</td>'

    body = []
    for t in table["rows"]:
        row     = t["row"]
        picked  = rate_pick == t["n"] - 1
        classes = " ".join(c for c, on in (("hull", row["supported"]), ("pick", picked)) if on)
        rate    = escape(_fmt_rate_range(row)) if row["supported"] else '<span class="dim">·</span>'
        changes = "".join(
            f'<span class="team"><b>T{team}</b>{escape(text)}</span>' for team, text in t["changes"]
        ) or '<span class="dim">—</span>'
        body.append(
            f'<tr class="{classes}" title="python plan_week.py --pick {t["n"]}">'
            f'<td class="n">{"▸ " if row["supported"] else ""}{t["n"]}</td>'
            f'<td>{row["port_ev"]:.1f}</td>'
            f'<td class="edge">{row["cost"]:.1f}</td>{signed(t["pts"])}{signed(t["budget"])}'
            f'<td>{row["diff_gain"]:.1f}</td>'
            f'<td class="cut" style="--w: {min(max(t["cut"], 0.0), 1.0):.0%}">{t["cut"]:.0%}</td>'
            f'<td class="edge">{_fmt_opt(t["lost"], ".1f")}</td><td>{_fmt_opt(t["gain"], ".1f")}</td>'
            f'<td class="edge">{t["turbos"]}</td>'
            + (f'<td class="neg">-{t["hits"]:g}</td>' if t["hits"] else '<td class="dim">—</td>')
            + f'<td class="left">{rate}{"<span class=tag>rate</span>" if picked else ""}</td>'
            f'<td class="left changes">{changes}</td></tr>'
        )

    legend = "".join(
        f"<dt>{escape(term)}</dt><dd>{escape(' '.join(text))}</dd>" for term, text in _legend(metric)
    )
    HTML_PATH.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Frontier</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<h1>EV vs differentiation frontier</h1>
<p class="meta">{escape("  ·  ".join(meta))}</p>
<p class="baseline">Baseline (row 1): <b>{table['top_pts']:.1f}</b> pts + <b>{table['top_budget']:.1f}</b> budget value
  &nbsp;|&nbsp; total shared EV <b>{table['total_shared']:.1f}</b></p>
{_baseline_html(table['rows'][0]['row']['results'], team_names)}
<table class="frontier">
<thead>
<tr>
  <th rowspan="2">#</th><th rowspan="2">Port EV</th>
  <th class="group edge" colspan="5">vs row 1</th>
  <th class="group edge" colspan="2">vs row above</th>
  <th class="edge" rowspan="2">Turbos</th><th rowspan="2">Hits</th>
  <th class="left" rowspan="2">Chosen if r</th><th class="left" rowspan="2">Changes vs row 1</th>
</tr>
<tr>
  <th class="edge">Cost</th><th>Pts now</th><th>Budget</th><th>Diff gain</th><th>Cut</th>
  <th class="edge">EV lost</th><th>Diff +</th>
</tr>
</thead>
<tbody>
{chr(10).join(body)}
</tbody>
</table>
<dl>{legend}</dl>
</body>
</html>
""")
    print(f"  Table saved to {HTML_PATH.name}")


def log_pick(settings: dict, row_no: int, row: dict) -> None:
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "logged_at", "gameday", "row", "cost", "diff_gain",
                "total_shared", "worst_pair", "port_ev", "r_lo", "r_hi", "supported",
            ])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            settings.get("gameday"),
            row_no,
            row["cost"],
            row["diff_gain"],
            row["total_shared"],
            row["worst_pair"],
            row["port_ev"],
            round(row["r_lo"], 4),
            "inf" if row["r_hi"] == float("inf") else round(row["r_hi"], 4),
            row["supported"],
        ])
    print(f"\n  Logged pick to {LOG_PATH.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Trace the portfolio EV vs differentiation frontier and pick a point."
    )
    parser.add_argument("csv_file", nargs="?", default=settings.get("data_file", "sample_data.csv"))
    parser.add_argument("--pts-per-1m", type=float, default=settings.get("pts_per_1m_per_race", 0.0), metavar="PTS")
    parser.add_argument("--remaining-races", type=int, default=settings.get("remaining_races", 23), metavar="N")
    parser.add_argument("--weights", type=float, nargs="+", default=settings.get("team_weights", None), metavar="W")
    parser.add_argument(
        "--step", type=float, default=10.0, metavar="PTS",
        help="Lower the total shared-EV cap by at least this much per solve (default 10). "
             "Smaller = finer frontier, more solves.",
    )
    parser.add_argument("--max-solves", type=int, default=60, metavar="N", help="Safety cap on solves (default 60)")
    parser.add_argument(
        "--max-transfers", type=int, default=None, metavar="N",
        help="Hard cap on transfers per team, e.g. 1 to model banking a free transfer",
    )
    parser.add_argument("--rate", type=float, default=None, metavar="R", help="Highlight the row this exchange rate picks")
    parser.add_argument("--pick", type=int, default=None, metavar="N", help="Print full transfer plans for frontier row N")
    parser.add_argument("--log", action="store_true", help=f"With --pick, append the choice to {LOG_PATH.name}")
    args = parser.parse_args()

    if args.step <= 0:
        parser.error("--step must be positive")

    budget_pts_weight = args.pts_per_1m * args.remaining_races
    xdelta_confidence = float(settings.get("xdelta_confidence", 1.0))
    metric = "net xPts + budget value" if budget_pts_weight else "net xPts"

    df = load_projections(args.csv_file)

    current_teams   = get_current_teams(settings)
    budgets         = [team_budget(t) for t in current_teams]
    limitless_teams = [t - 1 for t in settings.get("limitless", [])]
    wildcard_teams  = [t - 1 for t in settings.get("wildcard",  [])]
    limitless_set   = set(limitless_teams)

    solve_kwargs = dict(
        budgets=budgets,
        locked=[c.upper() for c in settings.get("locked", [])],
        banned=[c.upper() for c in settings.get("banned", [])],
        budget_pts_weight=budget_pts_weight,
        team_weights=args.weights,
        max_transfers=args.max_transfers,
        limitless_teams=limitless_teams,
        wildcard_teams=wildcard_teams,
        max_free_transfers=[t.get("free_transfers", 2) for t in current_teams],
    )

    if budget_pts_weight:
        print(f"  pts/1M/race : {args.pts_per_1m}  ×  {args.remaining_races} races  =  {budget_pts_weight:.1f} pts/M total")
    if args.max_transfers is not None:
        print(f"  max transfers per team : {args.max_transfers}")
    if limitless_set:
        print(f"  limitless   : {'+'.join(f'T{k+1}' for k in sorted(limitless_set))}  (shared EV measured vs pre-chip picks)")

    fingerprint = input_fingerprint(
        args.csv_file, current_teams, solve_kwargs,
        extra={"step": args.step, "max_solves": args.max_solves, "xdelta_confidence": xdelta_confidence},
    )

    if args.pick is None:
        print(f"\nSweeping total shared-EV cap (step {args.step:g})...")
        rows = run_frontier(
            df, current_teams, solve_kwargs, limitless_set,
            budget_pts_weight, xdelta_confidence, args.step, args.max_solves,
        )
        if not rows:
            print("\nNo feasible portfolio found.")
            sys.exit(1)

        frontier = pareto_filter(rows)
        add_rate_ranges(frontier)
        save_frontier(frontier, fingerprint)
        rate_pick = pick_for_rate(frontier, args.rate) if args.rate is not None else None
        table = frontier_table(frontier, len(current_teams), limitless_set)
        print_frontier(table, rate_pick, metric)
        meta = [
            f"solved {datetime.now():%a %d %b %H:%M}",
            f"gameday {settings.get('gameday')}",
            f"{budget_pts_weight:g} pts/M budget value",
            f"step {args.step:g}",
        ]
        if args.max_transfers is not None:
            meta.append(f"max {args.max_transfers} transfers per team")
        write_frontier_html(table, rate_pick, metric, meta, [t["name"] for t in current_teams])
        return

    frontier = load_frontier(fingerprint)
    if not 1 <= args.pick <= len(frontier):
        print(f"\nError: --pick must be between 1 and {len(frontier)}")
        sys.exit(1)

    row     = frontier[args.pick - 1]
    results = row["results"]
    print(f"\n\n{'=' * 60}")
    print(f"  ROW {args.pick}  —  cost {row['cost']:.1f} EV, diff gain {row['diff_gain']:.1f}, worst pair {row['worst_pair']:.1f}")
    print(f"  Implied exchange rate: {_fmt_rate_range(row)}")

    for i, (result, current_team, budget) in enumerate(zip(results, current_teams, budgets), start=1):
        print_transfer_plan(
            result, current_team, budget, team_idx=i,
            budget_pts_weight=budget_pts_weight, xdelta_confidence=xdelta_confidence,
        )

    max_count = max(
        compute_overlap(results[a], results[b])
        for a in range(len(results)) for b in range(a + 1, len(results))
    )
    print_portfolio_summary(
        results, current_teams, max_pairwise_overlap=max_count,
        budget_pts_weight=budget_pts_weight, xdelta_confidence=xdelta_confidence,
    )

    if args.log:
        log_pick(settings, args.pick, row)


if __name__ == "__main__":
    main()
