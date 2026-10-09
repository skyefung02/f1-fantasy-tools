"""
F1 Fantasy Transfer Solver
==========================
Integer linear programs choosing next race week's teams from your current ones.
Every team has:
  - Exactly 5 drivers and 2 constructors
  - Exactly 1 turbo driver (scores 2x points)
  - Total price <= budget
  - A points penalty for each transfer beyond its free transfers

solve_portfolio_transfers solves all teams jointly and is what plan_week.py uses.
solve_with_transfers solves one team and is used by diff_ev_plot.py.

Projections DataFrame columns:
  name, type (driver|constructor), price, expected_points, [xDeltaPrice]
"""

import sys
from pathlib import Path
import pandas as pd
import shutil
import pulp


def _cbc_solver():
    """Prefer a native CBC install (e.g. conda-forge coin-or-cbc) over PuLP's
    bundled binary, which is x86_64-only on macOS and fails on Apple Silicon."""
    cbc = Path(sys.executable).parent / "cbc"
    path = str(cbc) if cbc.exists() else shutil.which("cbc")
    if path:
        return pulp.COIN_CMD(path=path, msg=0)
    return pulp.PULP_CBC_CMD(msg=0)


_warned_missing_banned: set[str] = set()


def check_codes(locked: set[str], banned: set[str], all_codes: set[str]) -> set[str]:
    """Validate locked/banned codes against the data file.

    A locked code that isn't in the data is fatal: the constraint can never be
    satisfied, so silently dropping it would return a team that quietly ignores
    what you asked for.

    A banned code that isn't in the data is harmless - it's already unpickable -
    so it only warns. This happens routinely mid-season: an injured or replaced
    driver drops out of the projections while your ban list still names him.
    Returns the set of banned codes that were dropped.
    """
    missing_locked = locked - all_codes
    if missing_locked:
        raise ValueError(f"Locked codes not found in data: {sorted(missing_locked)}")

    missing_banned = banned - all_codes
    # Sweeps call the solver dozens of times; warn once per code per process.
    unreported = missing_banned - _warned_missing_banned
    if unreported:
        _warned_missing_banned.update(unreported)
        print(
            f"  note: banned {', '.join(sorted(unreported))} not in the data "
            f"(already unpickable, ban ignored)",
            file=sys.stderr,
        )
    return missing_banned



# ── Team comparison ───────────────────────────────────────────────────────────

def compute_differential_ev(team: dict, reference_tlas: set[str]) -> float:
    """
    Sum of base expected_points for assets in team that are NOT in reference_tlas.

    Uses base EV (turbo not doubled) since turbo is a team-specific choice
    independent of which assets are shared with the reference.

    Parameters
    ----------
    team           : result dict from solve() / solve_with_transfers() etc.
    reference_tlas : set of uppercase asset codes (drivers + constructors)
                     that define the reference team
    """
    ref = {t.upper() for t in reference_tlas}
    diff_ev = sum(
        d["expected_points"] for d in team["drivers"]
        if d["name"].upper() not in ref
    ) + sum(
        c["expected_points"] for c in team["constructors"]
        if c["name"].upper() not in ref
    )
    return round(diff_ev, 2)


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
    banned -= check_codes(locked, banned, all_codes)

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
    prob.solve(_cbc_solver())

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
    max_free_transfers: int | list[int] = 2,
    penalty_per_transfer: int = 10,
    budget_pts_weight: float = 0.0,
    team_weights: list[float] | None = None,
    max_transfers: int | None = None,
    limitless_teams: list[int] | None = None,
    wildcard_teams: list[int] | None = None,
    diff_ev_weight: float = 0.0,
    diff_ev_reference_tlas: set[str] | None = None,
    max_total_shared_ev: float | None = None,
) -> list[dict]:
    """
    Solve optimal transfers for all teams jointly in a single ILP.

    All teams are optimised simultaneously with pairwise overlap constraints
    enforced across every team pair at once. Unlike a sequential approach, no
    team is prioritised over others — the solver trades off freely between teams
    to maximise the weighted total portfolio score.

    Parameters
    ----------
    current_teams          : list of enrich_team() dicts from fetch_teams.py
    budgets                : available budget per team (total_price + budget_remaining)
    team_weights           : per-team objective weights (e.g. [0.2, 0.4, 0.4]).
                             Higher weight = solver prioritises that team more.
                             Defaults to equal weights [1.0, 1.0, ...] if None.
    max_transfers          : hard cap on total transfers per team (e.g. 1 to model
                             banking a free transfer). None = unconstrained.
    limitless_teams        : 0-indexed list of teams playing the Limitless chip.
                             Limitless teams are solved with unlimited budget, no
                             transfer penalty, and pure xPts (no xDeltaPrice).
                             Overlap is measured on the Limitless picks themselves —
                             the lineup that scores this week — so they are chosen
                             jointly with the other teams. The reported budget value
                             is that of the pre-chip lineup, which is what the team
                             reverts to and so whose price changes it experiences.
    wildcard_teams         : 0-indexed list of teams playing the Wildcard chip.
                             Wildcard teams keep their normal budget but pay no
                             transfer penalty (all transfers are free). Budget value
                             (xDeltaPrice) is still included in the objective.
    diff_ev_weight         : λ in [0, 1]. Blends the per-asset EV coefficient between
                             full weight (λ=0, pure total EV) and discounted for
                             reference assets (λ=1, pure differential EV).
                             Reference assets have their selection coefficient scaled
                             by (1 − λ); non-reference assets keep full weight.
                             Ignored when diff_ev_reference_tlas is None or empty.
    diff_ev_reference_tlas : Set of uppercase asset codes defining the reference team.
                             Typically loaded from settings.json reference_team_picks.
    max_total_shared_ev    : Cap on shared EV summed over every team pair. Shared EV
                             for a pair is the xPts-weighted version of overlap: each
                             shared driver or constructor contributes its xPts, and a
                             shared turbo contributes the driver's xPts again.
                             Negative xPts count as zero. None = no cap.
    """
    n_teams       = len(current_teams)
    limitless_set = set(limitless_teams or [])
    wildcard_set  = set(wildcard_teams  or [])
    locked  = {c.upper() for c in (locked or [])}
    banned  = {c.upper() for c in (banned or [])}

    mft_list = (
        max_free_transfers if isinstance(max_free_transfers, list)
        else [max_free_transfers] * n_teams
    )

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
    banned   -= check_codes(locked, banned, all_codes)
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

    # ── Differential EV coefficients ─────────────────────────────────────────
    # When diff_ev_weight (λ) > 0, reference assets have their selection
    # coefficient scaled by (1 − λ) while non-reference assets keep weight 1.
    # Effective objective per asset:
    #   non-ref driver selection : 1.0 × ev    (unique to this team → full credit)
    #   ref     driver selection : (1−λ) × ev  (shared with reference → discounted)
    #   turbo bonus              : same scale as corresponding selection coefficient
    #   xDeltaPrice              : always full weight (budget value is independent
    #                              of weekly differential EV concept)
    ref = {t.upper() for t in (diff_ev_reference_tlas or set())}
    use_diff = bool(ref) and diff_ev_weight > 0.0

    def _d_coeff(i: int) -> float:
        if use_diff and drivers.loc[i, "name"].upper() in ref:
            return 1.0 - diff_ev_weight
        return 1.0

    def _c_coeff(j: int) -> float:
        if use_diff and constructors.loc[j, "name"].upper() in ref:
            return 1.0 - diff_ev_weight
        return 1.0

    # ── Objective: weighted sum of (score − penalty) across all teams ─────────
    obj_terms = []
    for k in range(n_teams):
        w     = team_weights[k]
        score = pulp.lpSum(
            (x_d[k][i] + t_d[k][i]) * _d_coeff(i) * drivers.loc[i, "expected_points"]
            for i in range(n_d)
        ) + pulp.lpSum(
            x_c[k][j] * _c_coeff(j) * constructors.loc[j, "expected_points"]
            for j in range(n_c)
        )
        # Limitless teams optimise pure xPts only — their picks revert next week,
        # so the price changes they experience are the pre-chip lineup's, whatever
        # is picked here.
        if has_delta and k not in limitless_set:
            score += pulp.lpSum(
                x_d[k][i] * drivers.loc[i, "xDeltaPrice"] * budget_pts_weight for i in range(n_d)
            ) + pulp.lpSum(
                x_c[k][j] * constructors.loc[j, "xDeltaPrice"] * budget_pts_weight for j in range(n_c)
            )
        obj_terms.append(w * (score - penalty_per_transfer * e[k]))
    prob += pulp.lpSum(obj_terms)

    # ── Per-team standard + transfer constraints ───────────────────────────────
    for k, (team, budget) in enumerate(zip(current_teams, budgets)):
        is_limitless    = k in limitless_set
        is_wildcard     = k in wildcard_set
        eff_budget      = 999.0 if is_limitless else budget
        eff_mft         = 7     if (is_limitless or is_wildcard) else mft_list[k]

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
            <= eff_budget,
            f"k{k}_budget",
        )

        current_d = {d["tla"].upper() for d in team["drivers"]}
        current_c = {c["tla"].upper() for c in team["constructors"]}
        kept = (
            pulp.lpSum(x_d[k][i] for i in range(n_d) if drivers.loc[i, "name"].upper() in current_d)
            + pulp.lpSum(x_c[k][j] for j in range(n_c) if constructors.loc[j, "name"].upper() in current_c)
        )
        prob += e[k] >= 7 - kept - eff_mft, f"k{k}_excess_lb"
        if max_transfers is not None and not is_limitless and not is_wildcard:
            prob += kept >= 7 - max_transfers, f"k{k}_hard_max_xfers"

    # ── Pairwise overlap constraints (all pairs, jointly) ─────────────────────
    # Overlap between teams a and b = shared driver picks + shared constructor
    # picks + same turbo. Requires auxiliary binary vars to linearise AND of
    # two binary variables: z = x_a AND x_b  ⟺  z≤x_a, z≤x_b, z≥x_a+x_b-1
    #
    # Limitless teams are treated like any other: overlap is about the lineups
    # that score this week, and a Limitless team scores with its chip picks.
    shared_ev_terms = []   # one xPts-weighted shared expression per pair
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

        if max_total_shared_ev is not None:
            # Floored at zero: sharing a negative-xPts asset must not buy headroom
            shared_ev_terms.append(
                pulp.lpSum((z_d[i] + w_t[i]) * max(drivers.loc[i, "expected_points"], 0.0) for i in range(n_d))
                + pulp.lpSum(z_c[j] * max(constructors.loc[j, "expected_points"], 0.0) for j in range(n_c))
            )

    if max_total_shared_ev is not None and shared_ev_terms:
        prob += pulp.lpSum(shared_ev_terms) <= max_total_shared_ev, "total_shared_ev"

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(_cbc_solver())

    if prob.status != 1:
        raise RuntimeError(
            f"Solver did not find an optimal solution. Status: {pulp.LpStatus[prob.status]}"
        )

    # ── Parse one result dict per team ────────────────────────────────────────
    has_delta_d = "xDeltaPrice" in drivers.columns
    has_delta_c = "xDeltaPrice" in constructors.columns
    x_delta     = dict(zip(df["name"].str.upper(), df["xDeltaPrice"])) if "xDeltaPrice" in df.columns else {}
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
        is_limitless = k in limitless_set
        is_wildcard  = k in wildcard_set
        # Limitless picks revert next week — the team pays no transfer penalty and
        # experiences the price changes of its pre-chip lineup, not of these picks.
        # Wildcard keeps normal budget/picks but pays no transfer penalty.
        if is_limitless:
            budget_value = sum(x_delta.get(tla, 0.0) for tla in current_d | current_c) * budget_pts_weight
        else:
            budget_value = (
                sum(d["xDeltaPrice"] for d in sel_drivers)
                + sum(c["xDeltaPrice"] for c in sel_constructors)
            ) * budget_pts_weight

        driver_out  = sorted(current_d - new_d_tlas)
        driver_in   = sorted(new_d_tlas - current_d)
        constr_out  = sorted(current_c - new_c_tlas)
        constr_in   = sorted(new_c_tlas - current_c)
        n_transfers = len(driver_out) + len(constr_out)
        penalty_pts = 0 if (is_limitless or is_wildcard) else max(0, n_transfers - mft_list[k]) * penalty_per_transfer

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
            "is_limitless":              is_limitless,
            "is_wildcard":               is_wildcard,
            "driver_transfers_out":      driver_out,
            "driver_transfers_in":       driver_in,
            "constructor_transfers_out": constr_out,
            "constructor_transfers_in":  constr_in,
        })

    return results
