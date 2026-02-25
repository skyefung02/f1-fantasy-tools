"""
F1 Fantasy Team Fetcher
=======================
Fetches your 3 picked teams from the F1 Fantasy website.

See README.md for full setup instructions. Quick summary:

  1. Log into fantasy.formula1.com in Chrome
  2. DevTools (Cmd+Option+I) → Network → Fetch/XHR → hard-refresh (Cmd+Shift+R)
     on https://fantasy.formula1.com/en/my-team
  3. Copy the Cookie header from any request → paste into .env:
         F1_FANTASY_COOKIE=<paste here>
  4. Copy the UUID from the getteam URL → add to settings.json:
         "user_uuid": "<your uuid>"

Usage
-----
    python fetch_teams.py
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import requests

SETTINGS_FILE = Path(__file__).parent / "settings.json"
ENV_FILE      = Path(__file__).parent / ".env"
BASE_URL      = "https://fantasy.formula1.com"

DEFAULT_SETTINGS = {
    "budget": 100.0,
    "data_file": "sample_data.csv",
    "locked": [],
    "banned": [],
    "top_n": 5,
    "gameday": 1,
    "user_uuid": None,
}

SLOT_LABELS = {1: "CURRENT RACE", 2: "NEXT RACE", 3: "SLOT 3"}
CONSTRUCTOR_POSITIONS = {6, 7}


# ── Settings / env ────────────────────────────────────────────────────────────

def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    return DEFAULT_SETTINGS


def load_cookie() -> str | None:
    cookie = os.environ.get("F1_FANTASY_COOKIE", "").strip()
    if cookie:
        return cookie
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("F1_FANTASY_COOKIE="):
                return line[len("F1_FANTASY_COOKIE="):].strip()
    return None


# ── API client ────────────────────────────────────────────────────────────────

class F1FantasyClient:
    def __init__(self, cookie: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Cookie": cookie,
            "Accept": "application/json",
        })

    def _get(self, path: str, params: dict = None) -> dict:
        resp = self.session.get(f"{BASE_URL}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_players(self, gameday: int = 1) -> dict[str, dict]:
        """Fetch all players (drivers + constructors) keyed by PlayerId string."""
        buster = int(time.time() * 1000)
        data = self._get(f"/feeds/drivers/{gameday}_en.json", params={"buster": buster})
        players = data.get("Data", {}).get("Value", [])
        return {str(p["PlayerId"]): p for p in players}

    def get_my_teams(self, user_uuid: str, gameday: int = 1) -> list[dict]:
        """Fetch all picked teams for the authenticated user."""
        buster = int(time.time() * 1000)
        data = self._get(
            f"/services/user/gameplay/{user_uuid}/getteam"
            f"/{gameday}/{gameday}/{gameday}/{gameday}",
            params={"buster": buster},
        )
        return data.get("Data", {}).get("Value", {}).get("userTeam", [])


# ── Data enrichment ───────────────────────────────────────────────────────────

def enrich_team(raw_team: dict, players: dict[str, dict]) -> dict:
    """Cross-reference picked player IDs with full player data."""
    drivers = []
    constructors = []
    turbo_driver = None

    for pick in raw_team.get("playerid", []):
        pid = str(pick["id"])
        player = players.get(pid, {})
        is_constructor = pick.get("playerpostion") in CONSTRUCTOR_POSITIONS
        is_turbo = bool(pick.get("iscaptain"))

        name = player.get("FUllName") or f"Player #{pid}"
        entry = {
            "name": name,
            "tla": player.get("DriverTLA", ""),
            "price": float(player.get("Value", 0)),
            "season_score": int(player.get("OverallPpints") or 0),
            "is_turbo": is_turbo,
        }

        if is_constructor:
            constructors.append(entry)
        else:
            if is_turbo:
                turbo_driver = name
            drivers.append(entry)

    drivers.sort(key=lambda d: d["is_turbo"])

    total_price = round(
        sum(d["price"] for d in drivers) + sum(c["price"] for c in constructors), 1
    )

    return {
        "slot": raw_team.get("teamno"),
        "name": unquote(raw_team.get("teamname", "")),
        "budget_remaining": float(raw_team.get("teambal", 0)),
        "drivers": drivers,
        "turbo_driver": turbo_driver,
        "constructors": constructors,
        "total_price": total_price,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_team(team: dict) -> None:
    sep = "─" * 60
    label = SLOT_LABELS.get(team["slot"], f"SLOT {team['slot']}")
    print(f"\n{'':=<60}")
    print(f"  SLOT {team['slot']}  —  {label}")
    print(f"  {team['name']}")
    print(f"{'':=<60}")

    print(f"\n  DRIVERS")
    print(f"  {'Name':<28} {'TLA':<6} {'Price':>5}  {'Season Pts':>10}")
    print(f"  {'─'*28} {'─'*6} {'─'*5}  {'─'*10}")
    for d in team["drivers"]:
        tag = " ★ TURBO" if d["is_turbo"] else ""
        print(f"  {d['name']:<28} {d['tla']:<6} {d['price']:>5.1f}  {d['season_score']:>10}{tag}")

    print(f"\n  CONSTRUCTORS")
    print(f"  {'Name':<28} {'':6} {'Price':>5}  {'Season Pts':>10}")
    print(f"  {'─'*28} {'─'*6} {'─'*5}  {'─'*10}")
    for c in team["constructors"]:
        print(f"  {c['name']:<28} {'':6} {c['price']:>5.1f}  {c['season_score']:>10}")

    print(f"\n  Total spent:     {team['total_price']:.1f}")
    print(f"  Budget remaining:{team['budget_remaining']:.1f}")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    settings = load_settings()

    cookie = load_cookie()
    if not cookie:
        print("Error: F1_FANTASY_COOKIE is not set.")
        print("Add it to a .env file in this folder:  F1_FANTASY_COOKIE=<your cookie>")
        sys.exit(1)

    user_uuid = settings.get("user_uuid")
    if not user_uuid:
        print("Error: 'user_uuid' is not set in settings.json.")
        print("Find it in the Network tab: the UUID in the getteam/getusergamedaysv1 URL.")
        print('Then add it:  "user_uuid": "<your uuid>"')
        sys.exit(1)

    gameday = settings.get("gameday", 1)
    client  = F1FantasyClient(cookie=cookie)

    print(f"Fetching player data (gameday {gameday})...")
    try:
        players = client.get_players(gameday=gameday)
    except requests.HTTPError as e:
        print(f"Error fetching players: {e}")
        sys.exit(1)

    print(f"Fetching your teams...")
    try:
        raw_teams = client.get_my_teams(user_uuid=user_uuid, gameday=gameday)
    except requests.HTTPError as e:
        print(f"Error fetching teams: {e}")
        print("Your cookie may be expired — re-extract it from the browser.")
        sys.exit(1)

    if not raw_teams:
        print("No teams found. Check that your cookie is valid and user_uuid is correct.")
        sys.exit(0)

    for raw in raw_teams:
        team = enrich_team(raw, players)
        print_team(team)


if __name__ == "__main__":
    main()
