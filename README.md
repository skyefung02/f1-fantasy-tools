# F1 Fantasy Tools

An integer linear programming (ILP) solver for F1 Fantasy team selection. Given a CSV of driver and constructor prices and expected points, it finds the optimal team that maximises total expected points within your budget.

## How it works

The solver uses binary decision variables and the CBC solver (via [PuLP](https://coin-or.github.io/pulp/)) to solve the following problem:

**Maximise:** `Σ (x_d[i] + t_d[i]) × pts_i  +  Σ x_c[j] × pts_j`

**Subject to:**
- Exactly 5 drivers selected
- Exactly 2 constructors selected
- Exactly 1 turbo driver (`t_d[i] ≤ x_d[i]`) — scores 2× points
- Total price ≤ budget
- Any locked picks must be selected
- Any banned picks cannot be selected

For top-N results, each solution is excluded from subsequent solves by constraining the sum of its 8 defining binary variables (5 drivers + 2 constructors + 1 turbo) to be ≤ 7, forcing at least one pick to differ.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python solver.py
```

All settings are read from `settings.json` by default. You can also override at the CLI:

```bash
python solver.py path/to/data.csv --budget 95 --top 3
```

## Settings

Edit `settings.json` to configure the solver without touching the CLI:

```json
{
    "budget": 100.0,
    "data_file": "sample_data.csv",
    "top_n": 5,
    "locked": ["NOR", "MCL"],
    "banned": ["ALO", "STR"]
}
```

| Key | Description |
|---|---|
| `budget` | Total spend cap (default `100.0`) |
| `data_file` | Path to the input CSV |
| `top_n` | Number of ranked solutions to return |
| `locked` | Picks that **must** appear in the team |
| `banned` | Picks that **cannot** appear in the team |

CLI flags (`--budget`, `--top`) override `settings.json` values.

## CSV format

The input CSV must have these columns:

| Column | Description |
|---|---|
| `type` | `driver` or `constructor` |
| `name` or `code` | Short identifier (e.g. `VER`, `MCL`) |
| `price` | Price in fantasy budget units |
| `expected_points` or `xPts` | Projected points for the round |

See [sample_data.csv](sample_data.csv) for an example.

## Example output

```
Settings loaded from: .../settings.json
  data_file : sample_data.csv
  budget    : 100.0
  top_n     : 5

           RANK #1  —  168.00 pts
──────────────────────────────────────────────────────

DRIVERS
  Name                       Price     Pts  Note
  ───────────────────────── ──────  ──────  ──────────
  BOT                          5.9     4.3
  LAW                          6.5     1.9
  COL                          6.2     0.9
  BOR                          6.4     0.9
  LEC                         22.8    44.6  (22.3) ★ TURBO (2x)

CONSTRUCTORS
  Name                       Price     Pts
  ───────────────────────── ──────  ──────
  MCL                         28.9    58.8
  FER                         23.3    56.6

──────────────────────────────────────────────────────
  Budget:           100.0
  Total spent:      100.0
  Remaining:        0.0
  Total points:     168.00  (incl. turbo bonus)
──────────────────────────────────────────────────────
```

## Fetching your teams from the website

`fetch_teams.py` pulls your current team selections directly from the F1 Fantasy website so you don't have to enter them manually.

### One-time setup

**Step 1 — Extract your session cookie**

The script authenticates using your browser's session cookie. You need to copy it once (and re-copy it when it expires, usually after a few days).

1. Log into [fantasy.formula1.com](https://fantasy.formula1.com) in Chrome
2. Open DevTools: `Cmd+Option+I`
3. Go to the **Network** tab and select the **Fetch/XHR** filter
4. Hard-refresh the My Team page: `Cmd+Shift+R`
5. Click any request to `fantasy.formula1.com` in the list
6. In the **Headers** panel on the right, find the **Request Headers** section
7. Copy the full value of the `Cookie:` header (it's a long string)
8. Create a file called `.env` in this project folder and paste it in like this — no quotes:

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
4. Add it to `settings.json`:
   ```json
   "user_uuid": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
   ```

### Usage

```bash
python fetch_teams.py
```

This prints all 3 of your team slots with drivers, constructors, turbo pick, prices, and remaining budget.

### Settings for fetch_teams.py

| Key | Description |
|---|---|
| `user_uuid` | Your account UUID (see setup above) |
| `gameday` | Game day number to fetch (default `1`) |

### Cookie expiry

If you get an authentication error, your cookie has expired. Repeat Step 1 above to get a fresh one and update `.env`.

## Files

| File | Description |
|---|---|
| `solver.py` | ILP solver, top-N logic, CLI |
| `fetch_teams.py` | Fetches your picked teams from the F1 Fantasy website |
| `settings.json` | Runtime configuration |
| `sample_data.csv` | Example driver/constructor data |
| `requirements.txt` | Python dependencies |
