"""
F1 Fantasy Avatars
==================
Draws a small cartoon avatar per driver and a team-colour car per constructor into
avatars/, which plan_week.py shows on the frontier page's team cards. Purely
cosmetic — delete the folder and the page simply goes back to plain text.

Driver faces come from DiceBear's Avataaars style (https://www.dicebear.com),
built from the handful of features in DRIVERS below. The shirt is the team
colour from projections_snapshot.json, so a driver who changes team only needs
this re-run. A driver missing from DRIVERS gets a plain colour badge with his code.

Usage
-----
    python make_avatars.py            # (re)draw everything
    python make_avatars.py HAM BOT    # just these
"""

import sys
import json
from pathlib import Path

import requests

HERE          = Path(__file__).parent
SNAPSHOT_FILE = HERE / "projections_snapshot.json"
AVATAR_DIR    = HERE / "avatars"
API_URL       = "https://api.dicebear.com/9.x/avataaars/svg"

SKIN = {"fair": "ffdbb4", "light": "edb98a", "tan": "d08b5b", "brown": "ae5d29"}
HAIR = {"black": "2c1b18", "dark": "4a312c", "brown": "724133", "blond": "b58143", "platinum": "ecdcbf"}

# code: (skin, hair style, hair colour, facial hair or None)
DRIVERS = {
    "ALB": ("light", "shortFlat",  "black",    None),
    "ALO": ("light", "shortFlat",  "black",    "beardLight"),
    "ANT": ("light", "shortCurly", "dark",     None),
    "BEA": ("fair",  "shortCurly", "dark",     None),
    "BOR": ("light", "shortCurly", "black",    None),
    "BOT": ("fair",  "shortFlat",  "platinum", "moustacheMagnum"),
    "COL": ("light", "shortWaved", "dark",     None),
    "GAS": ("light", "shortFlat",  "dark",     "beardLight"),
    "HAD": ("light", "shortWaved", "black",    None),
    "HAM": ("brown", "dreads01",   "black",    "beardLight"),
    "HUL": ("fair",  "shortFlat",  "blond",    None),
    "LAW": ("fair",  "shortCurly", "blond",    None),
    "LEC": ("light", "shortWaved", "dark",     None),
    "LIN": ("tan",   "shortCurly", "black",    None),
    "NOR": ("light", "shortCurly", "dark",     None),
    "OCO": ("light", "shortFlat",  "black",    None),
    "PER": ("light", "shortRound", "black",    None),
    "PIA": ("fair",  "shortWaved", "brown",    None),
    "RUS": ("fair",  "shortWaved", "brown",    None),
    "SAI": ("light", "shortWaved", "dark",     None),
    "STR": ("light", "shortWaved", "black",    None),
    "TSU": ("light", "shortFlat",  "black",    None),
    "VER": ("fair",  "shortFlat",  "blond",    None),
}


def team_colours() -> dict[str, tuple[str, str]]:
    """code -> (type, team colour), preferring the active seat when a driver has two."""
    entities = json.loads(SNAPSHOT_FILE.read_text())["entities"].values()
    colours = {}
    for entity in sorted(entities, key=lambda e: e["isActive"]):
        colours[entity["abbreviation"].upper()] = (entity["type"], entity["color"].lstrip("#"))
    return colours


def driver_svg(code: str, shirt: str) -> str:
    skin, top, hair, facial = DRIVERS[code]
    params = {
        "seed": code,
        "backgroundColor": "transparent",
        "skinColor": SKIN[skin],
        "top": top,
        "hairColor": HAIR[hair],
        "facialHair": facial or "beardLight",
        "facialHairColor": HAIR[hair],
        "facialHairProbability": 100 if facial else 0,
        "accessoriesProbability": 0,
        "clothing": "shirtCrewNeck",
        "clothesColor": shirt,
        "eyes": "default",
        "eyebrows": "defaultNatural",
        "mouth": "default",
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def _luma(colour: str) -> float:
    r, g, b = (int(colour[i:i + 2], 16) for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def car_svg(colour: str) -> str:
    """Side-on F1 car in the team colour. Grey outlines keep white and dark liveries visible on either theme."""
    line = "57606a" if _luma(colour) > 200 else "8b949e"
    edge = f'stroke="#{line}" stroke-width="1.3" stroke-linejoin="round"'
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f'<g fill="#{colour}" {edge}>'
        '<path d="M5 31h14v7h-4l1 13h-6l-2-13H5z"/>'                                # rear wing
        '<path d="M9 61l4-11 17-2 8-12h9l4 8 11 2 12 6 20 8v4z"/>'                  # body, airbox to nose
        '<path d="M82 62h14v4H82z"/>'                                               # front wing
        '</g>'
        '<path d="M48 44q7-8 14 2" fill="none" stroke="#30363d" stroke-width="2.4" stroke-linecap="round"/>'   # halo
        '<g fill="#24292f" stroke="#8b949e" stroke-width="1.6">'
        '<circle cx="26" cy="61" r="11"/><circle cx="75" cy="61" r="11"/></g>'
        '<g fill="#8b949e"><circle cx="26" cy="61" r="3.5"/><circle cx="75" cy="61" r="3.5"/></g>'
        '</svg>'
    )


def badge_svg(code: str, colour: str) -> str:
    # Dark text on light team colours (Haas is white), light text otherwise
    ink = "1f2328" if _luma(colour) > 150 else "ffffff"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f'<rect width="100" height="100" fill="#{colour}"/>'
        f'<text x="50" y="62" text-anchor="middle" font-family="system-ui, sans-serif" '
        f'font-size="34" font-weight="700" fill="#{ink}">{code}</text></svg>'
    )


def main():
    colours = team_colours()
    wanted  = [c.upper() for c in sys.argv[1:]] or sorted(colours)
    AVATAR_DIR.mkdir(exist_ok=True)

    for code in wanted:
        if code not in colours:
            print(f"  {code}: not in {SNAPSHOT_FILE.name}, skipped")
            continue
        kind, colour = colours[code]
        if kind == "constructor":
            drawn, svg = "car", car_svg(colour)
        elif code in DRIVERS:
            drawn, svg = "face", driver_svg(code, colour)
        else:
            drawn, svg = "badge", badge_svg(code, colour)
        (AVATAR_DIR / f"{code}.svg").write_text(svg)
        print(f"  {code}: {drawn}  #{colour}")


if __name__ == "__main__":
    main()
