"""
F1 Fantasy Projection Fetcher
=============================
Pulls the analyst projections off f1fantasytools.com/team-calculator and writes
them straight into the CSV the solver reads, so sample_data.csv never has to be
typed in by hand again.

The team calculator is a Next.js app that server-renders its whole dataset into
the page: prices, ownership, per-race results, and the analyst simulation
(expected points, price-change probabilities, DNF odds, ...). No browser, no
API key and no login needed - one GET of the HTML has everything.

    xPts        <- analystSims[*].{drivers,constructors}.pts
    price       <- the driver/constructor entity list
    xDeltaPrice <- expectation of the price_change_probability distribution
                   (buckets pm7..pp15 are tenths of a million, values are %)

The site's other simulations (Classic Average, Weighted Average, Equal PPM, ...)
are computed in the browser from each entity's raceResults, so they are not
fetched here - only the uploaded analyst sims are.

Usage
-----
    python fetch_projections.py                 # write settings["data_file"]
    python fetch_projections.py --list-sims     # show what's published
    python fetch_projections.py --sim rhter_1   # pick a specific sim
    python fetch_projections.py --dry-run       # print, don't write
"""

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE           = Path(__file__).parent
SETTINGS_FILE  = HERE / "settings.json"
SNAPSHOT_FILE  = HERE / "projections_snapshot.json"
PAGE_URL       = "https://f1fantasytools.com/team-calculator"
USER_AGENT     = "f1-fantasy-tools/1.0 (personal fantasy solver; +local script)"

# Price-change buckets are named pm<N>/pp<N> where N is the move in tenths of a
# million; the value attached is a probability in percent.
PRICE_STEP = 0.1

CSV_COLUMNS = ["type", "code", "price", "xDeltaPrice", "xPts"]


# --------------------------------------------------------------------------- #
# Fetching / parsing
# --------------------------------------------------------------------------- #

def fetch_page(url=PAGE_URL, timeout=30):
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def flight_payload(html):
    """Reassemble the React server-component payload from the inline scripts.

    The page emits it as a series of self.__next_f.push([1, "<chunk>"]) calls;
    concatenating the string chunks gives one long (mostly JSON) blob.
    """
    decoder = json.JSONDecoder()
    chunks = []
    for match in re.finditer(r"self\.__next_f\.push\(", html):
        start = html.find("[", match.end() - 1)
        if start == -1:
            continue
        try:
            pushed, _ = decoder.raw_decode(html, start)
        except ValueError:
            continue
        if isinstance(pushed, list) and len(pushed) > 1 and isinstance(pushed[1], str):
            chunks.append(pushed[1])
    if not chunks:
        raise RuntimeError("no __next_f payload found - the site's markup changed")
    return "".join(chunks)


def _decode_at(flight, index):
    try:
        value, _ = json.JSONDecoder().raw_decode(flight, index)
    except ValueError:
        return None
    return value


def parse_entities(flight):
    """Every driver and constructor, keyed by id (e.g. RED_VER, MCL)."""
    entities = {}
    for match in re.finditer(r'\[\{"id":"', flight):
        array = _decode_at(flight, match.start())
        if not isinstance(array, list) or not array:
            continue
        first = array[0]
        if not isinstance(first, dict) or "abbreviation" not in first or "price" not in first:
            continue
        for entity in array:
            entities[entity["id"]] = entity
    if not entities:
        raise RuntimeError("no driver/constructor list found - the site's markup changed")
    return entities


def parse_analyst_sims(flight):
    key = '"analystSims":'
    index = flight.find(key)
    if index == -1:
        raise RuntimeError("no analystSims found - the site's markup changed")
    sims = _decode_at(flight, index + len(key))
    if not isinstance(sims, list):
        raise RuntimeError("analystSims was not a list - the site's markup changed")
    return sims


def parse_meta(flight):
    """Race-week context the solver's settings.json also cares about."""
    meta = {}
    index = flight.find('"nextRace":')
    if index != -1:
        race = _decode_at(flight, index + len('"nextRace":'))
        if isinstance(race, dict):
            meta["next_race"] = {
                k: race.get(k)
                for k in ("id", "name", "roundNumber", "sprint", "countryName", "start_times")
            }
    index = flight.find('"numberOfFutureRaces":')
    if index != -1:
        count = _decode_at(flight, index + len('"numberOfFutureRaces":'))
        if isinstance(count, int):
            meta["number_of_future_races"] = count
    return meta


# --------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------- #

def expected_price_change(distribution):
    """Expectation of a pm/pp bucket distribution, in millions."""
    total = 0.0
    for bucket, percent in (distribution or {}).items():
        if bucket.startswith("pm"):
            sign = -1
        elif bucket.startswith("pp"):
            sign = 1
        else:
            continue
        try:
            steps = int(bucket[2:])
        except ValueError:
            continue
        total += sign * steps * PRICE_STEP * (percent / 100.0)
    return total


def pick_sim(sims, wanted=None):
    if wanted:
        for sim in sims:
            if sim.get("id") == wanted or sim.get("name") == wanted:
                return sim
        raise SystemExit(
            f"no sim matching {wanted!r}; run --list-sims to see what's published"
        )
    active = [s for s in sims if s.get("active")] or sims
    return max(active, key=lambda s: s.get("date_uploaded") or "")


def build_rows(sim, entities):
    """One CSV row per projected entity, drivers first then constructors."""
    rows = []
    skipped = []
    for kind, csv_type in (("drivers", "driver"), ("constructors", "constructor")):
        section = sim.get(kind) or {}
        points = section.get("pts") or {}
        price_changes = section.get("price_change_probability") or {}
        entries = []
        for entity_id, x_pts in points.items():
            entity = entities.get(entity_id)
            if entity is None:
                skipped.append(f"{entity_id} (no price on the page)")
                continue
            if not entity.get("isActive", True):
                skipped.append(f"{entity_id} (inactive)")
                continue
            entries.append(
                {
                    "type": csv_type,
                    "id": entity_id,
                    "code": entity.get("abbreviation") or entity_id,
                    "price": float(entity["price"]),
                    "xDeltaPrice": round(expected_price_change(price_changes.get(entity_id)), 2),
                    "xPts": round(float(x_pts), 2),
                }
            )
        rows.extend(sorted(entries, key=lambda r: r["code"]))

    # An abbreviation can be ambiguous during a mid-season driver swap (both
    # RED_LAW and VRB_LAW exist). Fall back to the full id so the solver, which
    # keys on `code`, never sees a duplicate.
    seen = {}
    for row in rows:
        seen.setdefault(row["code"], []).append(row)
    for code, group in seen.items():
        if len(group) > 1:
            for row in group:
                row["code"] = row["id"]

    return rows, skipped


def write_csv(rows, path, backup=True):
    path = Path(path)
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print(f"{'type':<12}{'code':<8}{'price':>8}{'xDeltaPrice':>13}{'xPts':>9}")
    for row in rows:
        print(
            f"{row['type']:<12}{row['code']:<8}{row['price']:>8.1f}"
            f"{row['xDeltaPrice']:>13.2f}{row['xPts']:>9.2f}"
        )


# --------------------------------------------------------------------------- #

def load_settings():
    if SETTINGS_FILE.exists():
        with SETTINGS_FILE.open() as handle:
            return json.load(handle)
    return {}


def update_settings(meta):
    """Write the race-week numbers into settings.json.

    Edited with a regex rather than json.dump so the file keeps its hand-written
    formatting (and any comments a future format might allow) byte-for-byte -
    only the two numbers move. Returns a list of (key, old, new) for reporting.
    """
    wanted = {}
    round_number = (meta.get("next_race") or {}).get("roundNumber")
    if isinstance(round_number, int):
        wanted["gameday"] = round_number
    future_races = meta.get("number_of_future_races")
    if isinstance(future_races, int):
        wanted["remaining_races"] = future_races
    if not wanted or not SETTINGS_FILE.exists():
        return []

    text = SETTINGS_FILE.read_text()
    changed = []
    for key, value in wanted.items():
        pattern = re.compile(rf'("{key}"\s*:\s*)(-?\d+(?:\.\d+)?)')
        match = pattern.search(text)
        if match is None:
            print(f"  note: no {key!r} key in settings.json, skipped", file=sys.stderr)
            continue
        previous = match.group(2)
        if previous == str(value):
            continue
        text = text[: match.start()] + f"{match.group(1)}{value}" + text[match.end() :]
        changed.append((key, previous, value))

    if changed:
        SETTINGS_FILE.write_text(text)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sim", help="analyst sim id or name (default: newest active)")
    parser.add_argument("--list-sims", action="store_true", help="list published sims and exit")
    parser.add_argument("--out", help="output CSV (default: settings.json data_file)")
    parser.add_argument("--dry-run", action="store_true", help="print instead of writing")
    parser.add_argument("--no-backup", action="store_true", help="don't keep a .bak of the old CSV")
    parser.add_argument(
        "--no-settings",
        action="store_true",
        help="don't sync gameday/remaining_races into settings.json",
    )
    parser.add_argument("--url", default=PAGE_URL, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        html = fetch_page(args.url)
    except requests.RequestException as exc:
        sys.exit(f"could not reach {args.url}: {exc}")

    flight = flight_payload(html)
    sims = parse_analyst_sims(flight)

    if args.list_sims:
        for sim in sims:
            flag = "*" if sim.get("active") else " "
            print(
                f"{flag} {sim.get('id'):<14} {sim.get('name','')!r} "
                f"by {sim.get('analyst')}  race week {sim.get('raceweek')}  "
                f"uploaded {sim.get('date_uploaded')}"
            )
            note = sim.get("new_info_description")
            if note:
                print(f"    {note.strip()}")
        return

    entities = parse_entities(flight)
    meta = parse_meta(flight)
    sim = pick_sim(sims, args.sim)
    rows, skipped = build_rows(sim, entities)

    out_path = Path(args.out or load_settings().get("data_file", "sample_data.csv"))
    if not out_path.is_absolute():
        out_path = HERE / out_path

    print(
        f"{sim.get('name','?')} ({sim.get('id')}) by {sim.get('analyst')} - "
        f"race week {sim.get('raceweek')}, uploaded {sim.get('date_uploaded')}"
    )
    next_race = meta.get("next_race") or {}
    if next_race:
        sprint = " (sprint)" if next_race.get("sprint") else ""
        print(
            f"next race: round {next_race.get('roundNumber')} {next_race.get('name')}{sprint}"
            f"   races remaining: {meta.get('number_of_future_races')}"
        )
    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(skipped)}")

    if args.dry_run:
        print_table(rows)
        print(f"\n(dry run - {out_path} untouched)")
        return

    write_csv(rows, out_path, backup=not args.no_backup)
    with SNAPSHOT_FILE.open("w") as handle:
        json.dump(
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source": args.url,
                "meta": meta,
                "sim": sim,
                "entities": entities,
            },
            handle,
            indent=2,
        )

    if not args.no_settings:
        for key, previous, value in update_settings(meta):
            print(f"settings.json: {key} {previous} -> {value}")

    drivers = sum(1 for r in rows if r["type"] == "driver")
    print(
        f"wrote {len(rows)} rows ({drivers} drivers, {len(rows) - drivers} constructors) "
        f"to {out_path}"
    )
    print(f"snapshot -> {SNAPSHOT_FILE.name}")


if __name__ == "__main__":
    main()
