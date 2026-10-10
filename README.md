# F1 Fantasy Tools

A weekly planner for running three F1 Fantasy teams as a portfolio. Each race week it takes analyst projections and your current teams, and shows what it costs in expected points to make the three teams less alike, so you can choose the trade-off and get the transfers for it.

`plan_week.py` is the main tool. Everything else either feeds it or is a side tool.

## Setup

```bash
conda env create -f environment.yml
conda activate f1-fantasy
cp settings.example.json settings.json
```

`settings.json` is yours and is not tracked by git. Fill in `user_uuid` and add your session cookie to `.env`; both are covered in [Fetching your teams from the website](#fetching-your-teams-from-the-website).

## Gameweek workflow

**Step 1 — refresh projections and race-week numbers**

```bash
python fetch_projections.py
```

This pulls the latest analyst projections off `f1fantasytools.com/team-calculator` and rewrites your data file (`settings.json` → `data_file`) with fresh `price`, `xPts` and `xDeltaPrice` for every active driver and constructor. The previous file is kept as `.bak`, and the raw simulation is saved to `projections_snapshot.json`.

It also syncs two numbers into `settings.json` so you never have to work them out by hand:

| Setting | Source |
|---|---|
| `gameday` | the site's `nextRace.roundNumber` |
| `remaining_races` | the site's `numberOfFutureRaces` |

**This has to run before `fetch_teams.py`**, because both F1 Fantasy endpoints take `gameday` as a path parameter — with a stale value you'd silently snapshot the *previous* week's teams. Pass `--no-settings` to skip the sync and manage those two by hand.

Analysts re-upload as the weekend develops (practice, qualifying), so it's worth re-running before you lock in transfers. Add `--dry-run` to preview without writing, `--list-sims` to see what's published, or `--sim <id>` to pin a specific analyst's simulation.

If you'd rather enter numbers by hand, just edit `sample_data.csv` (or your data file) directly instead — see [CSV format](#csv-format).

**Mid-season seat swaps.** A driver's TLA stops being a unique key when he changes team mid-season: after Lawson moved VCARB → Red Bull the site carried both `VRB_LAW` (inactive, the seat you may still own) and `RED_LAW` (active, the one being projected). Only active entities are projected, so the single `LAW` row in the CSV is Red Bull's — and any tool matching on TLA will value your inactive holding at the *other* seat's price and points, silently missing the −35 you take for holding an asset through a race it doesn't contest.

`fetch_projections.py` detects this and warns:

```
WARNING: seat swap on LAW. You may still hold VRB_LAW (10.3m, inactive), but the
only LAW row in the CSV is RED_LAW (14.5m, active). ... add LAW to "banned" to
force it out (keeping it through the race costs -35).
```

Add the TLA to `banned` for that week. It also warns in reverse once the swap resolves, so a temporary ban doesn't quietly keep blocking a good pick:

```
note: TSU is banned but is active and unambiguous this week (projected 2.02 xPts)
      - if that ban was for a seat swap, it can go
```

**Step 2 — snapshot your teams (before touching the website)**

```bash
python fetch_teams.py --save
```

This saves your current pre-transfer teams to `teams_snapshot.json`. The other tools load from this snapshot instead of calling the live API, so they always see your teams as they were at the start of the week — not after you've already made transfers.

Run this once per gameweek, before making any changes on the F1 Fantasy website. The snapshot is overwritten the next time you run `--save`.

**Step 3 — set this week's state in `settings.json`**

Check `free_transfers` (one number per team), and set `limitless` or `wildcard` to the team numbers playing a chip, e.g. `"limitless": [1]`. Leave them as `[]` otherwise.

**Step 4 — solve the frontier**

```bash
python plan_week.py
```

This prints a table and writes the same table to `plan_week_frontier.html`, which is easier to keep open while you work. Each row is a complete plan for all three teams.

**Step 5 — pick a row**

```bash
python plan_week.py --pick 4          # full transfer plans for row 4
python plan_week.py --pick 4 --log    # ...and record the choice in plan_week_log.csv
```

`--pick` reads the frontier saved by the last table run, so it is instant. It refuses if the projections, teams, settings or flags have changed since then; re-run the table first.

## Reading the frontier

Row 1 is the highest-EV portfolio with no regard for differentiation. Every row below gives up some EV to make the teams overlap less.

Overlap is measured as **shared EV**: for each pair of teams, the xPts of every driver and constructor both hold, plus the turbo driver's xPts again if both turbo the same driver. It is summed over all three pairs. Assets with negative xPts count as zero.

| Column | Meaning |
|---|---|
| Port EV | Net xPts plus budget value, summed over the teams |
| Cost | Port EV given up against row 1, split into points this week (Pts now) and budget value (Budget) |
| Diff gain | Shared EV removed against row 1. Cut is the same figure as a share of row 1's shared EV |
| vs row above | EV lost and differentiation gained by moving down one row |
| Turbos | Distinct turbo drivers across the portfolio |
| Hits | Transfer penalty points taken |
| Chosen if r | The exchange rates (EV per point of differentiation) at which this row is the best choice. Rows marked `·` are real options that no single rate selects |
| Changes | Lineup swaps against row 1; `*` marks the turbo |

Budget value is each asset's expected price change, converted to points at `pts_per_1m_per_race` × `remaining_races`.

Useful flags:

| Flag | Effect |
|---|---|
| `--rate R` | Highlight the row an exchange rate of R selects |
| `--step PTS` | Lower the shared-EV cap by at least this much per solve (default 10). Smaller gives a finer frontier and more solves |
| `--max-transfers N` | Hard cap on transfers per team, e.g. `1` to see what banking a free transfer costs |
| `--weights W W W` | Weight the teams unequally in the objective |

### Chips

- **Limitless:** the team has unlimited budget and no transfer penalty for the week. Its chip lineup is chosen jointly with the other teams and counts towards shared EV like any other lineup, so lower rows may give it a slightly weaker lineup to reduce overlap. Price changes on chip picks are never realised, so its budget value is fixed at the pre-chip lineup's.
- **Wildcard:** the team keeps its normal budget and pays no transfer penalty.

To judge whether a chip is worth playing, run the table with and without it set and compare Port EV on row 1.

## Settings

| Key | Description |
|---|---|
| `data_file` | Path to the projections CSV |
| `user_uuid` | Your F1 Fantasy account UUID |
| `gameday` | Current race round — **auto-synced** by `fetch_projections.py`, don't hand-edit |
| `remaining_races` | Races left in the season — **auto-synced** by `fetch_projections.py`, don't hand-edit |
| `free_transfers` | Free transfers available to each team this week |
| `pts_per_1m_per_race` | Points you value 1M of budget at, per remaining race. `0` turns budget value off |
| `xdelta_confidence` | Discount applied to budget value when it is reported (`1` = none) |
| `locked` | Picks every team **must** hold |
| `banned` | Picks no team may hold |
| `limitless` / `wildcard` | Team numbers (1-indexed) playing that chip this week |
| `team_weights` | Optional per-team objective weights |
| `reference_team` | Team that `diff_ev_plot.py` measures differential EV against |

## CSV format

`fetch_projections.py` writes this file for you. It is gitignored, so a fresh clone won't have one until you run it.

| Column | Description |
|---|---|
| `type` | `driver` or `constructor` |
| `name` or `code` | Short identifier (e.g. `VER`, `MCL`) |
| `price` | Price in fantasy budget units |
| `expected_points` or `xPts` | Projected points for the round |
| `xDeltaPrice` | Expected price change after the round (optional) |

```csv
type,code,price,xDeltaPrice,xPts
driver,VER,27.6,0.11,24.5
driver,LEC,23.9,0.25,37.88
constructor,MCL,31.0,0.29,65.31
```

## Side tools

```bash
python wildcard_eval.py      # per team: normal transfers vs Wildcard, side by side
python diff_ev_plot.py       # total EV vs differential EV against a reference team
```

Neither is part of the weekly routine. `diff_ev_plot.py` enumerates many feasible lineups per team and plots them against `reference_team`, which is useful if you're chasing league rank more than points.

---

## Fetching your teams from the website

`fetch_teams.py` pulls your current team selections directly from the F1 Fantasy website so you don't have to enter them manually.

### One-time setup

**Step 1 — Extract your session cookie**

The script authenticates using your browser's session cookie. You need to copy it once (and re-copy it when it expires, usually after a few days).

1. Log into [fantasy.formula1.com](https://fantasy.formula1.com) in Chrome
2. Open DevTools: `Cmd+Option+I`
3. Go to the **Network** tab
4. In the filter/search box, type `buster` — this filters to the right request quickly
5. Click the `buster` request that appears
6. In the **Headers** panel on the right, find the **Request Headers** section
7. Find the `Cookie:` header — it starts with something like `consentUUID=...`
8. Copy the full value of the `Cookie:` header (it's a long string)
9. Create a file called `.env` in this project folder and paste it in like this — no quotes:

```
F1_FANTASY_COOKIE=<paste the full cookie string here>
```

**Step 2 — Find your user UUID**

Your UUID identifies your account in the API. It only needs to be set once.

1. After the hard-refresh in Step 1, look for a request in the Network tab whose URL contains `getteam` or `getusergamedaysv1`
2. The URL looks like:
   ```
   https://fantasy.formula1.com/services/user/gameplay/{uuid}/getteam/...
   ```
3. Copy the UUID (the part between `gameplay/` and `/getteam`)
4. Set it in `settings.json`:
   ```json
   "user_uuid": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
   ```

### Usage

```bash
python fetch_teams.py
```

This prints all 3 of your team slots with drivers, constructors, turbo pick, prices, and remaining budget.

### Cookie expiry

If you get an authentication error, your cookie has expired. Repeat Step 1 above to get a fresh one and update `.env`.

## Files

| File | Description |
|---|---|
| `plan_week.py` | Weekly planner: frontier table, HTML view and transfer plans |
| `solver.py` | The ILPs behind it (CBC via [PuLP](https://coin-or.github.io/pulp/)) |
| `fetch_projections.py` | Fetches xPts/price projections from f1fantasytools.com into your data file |
| `fetch_teams.py` | Fetches your teams from the F1 Fantasy website |
| `wildcard_eval.py`, `diff_ev_plot.py` | Side tools |
| `make_avatars.py`, `avatars/` | Cartoon driver avatars and team badges for the frontier page. Re-run after a driver changes team or joins the grid |
| `settings.example.json` | Template for `settings.json` |
| `settings.json` | Your configuration and this week's state (untracked) |
| `sample_data.csv` | Your working projections (generated, untracked) |
| `projections_snapshot.json` | Raw analyst simulation from the last `fetch_projections.py` run |
| `teams_snapshot.json` | Your teams as of the last `fetch_teams.py --save` |
| `environment.yml` | Conda environment |
