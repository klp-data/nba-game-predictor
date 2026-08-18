"""One colour per franchise, so a team looks the same on every chart.

Keyed by the franchise name, meaning the name a team carries today. Historical
names inherit from whatever the franchise became, so a 1990 Bullets row draws in
the Wizards colour. Keying on the franchise and not on the display name matters
for one case in particular: Hornets is both Charlotte today and what New Orleans
was called until 2013.

Mostly the official primaries. A handful are nudged because two teams share
almost the same official shade and a chart with both on it has to stay readable:
Raptors off Bulls red, Blazers and Rockets off Bulls red, Nets off Spurs black,
Jazz off Wizards navy, Pelicans off Nuggets navy, and the Knicks on their orange
rather than their blue, which keeps them apart from the Thunder.
"""
from __future__ import annotations

from src.plot_style import COLORS

FALLBACK = COLORS["neutral"]

TEAM_COLORS = {
    # East
    "Celtics": "#007A33",
    "Nets": "#4D4D4D",
    "Knicks": "#F58426",
    "76ers": "#ED174C",
    "Raptors": "#753BBD",
    "Bulls": "#CE1141",
    "Cavaliers": "#860038",
    "Pistons": "#1D42BA",
    "Pacers": "#FDBB30",
    "Bucks": "#00471B",
    "Hawks": "#E03A3E",
    "Hornets": "#1D1160",
    "Heat": "#98002E",
    "Magic": "#0077C0",
    "Wizards": "#002B5C",
    # West
    "Nuggets": "#0E2240",
    "Timberwolves": "#236192",
    "Thunder": "#007AC1",
    "Trail Blazers": "#A6192E",
    "Jazz": "#F9A01B",
    "Warriors": "#1D428A",
    "Clippers": "#C8102E",
    "Lakers": "#552583",
    "Suns": "#E56020",
    "Kings": "#5A2D81",
    "Mavericks": "#00538C",
    "Rockets": "#BA0C2F",
    "Grizzlies": "#5D76A9",
    "Pelicans": "#B4975A",
    "Spurs": "#000000",
}

# Names a franchise used to carry, pointing at what it is called now. Anything
# not in here and not a current nickname falls back to grey.
HISTORICAL = {
    "Packers": "Wizards",
    "Zephyrs": "Wizards",
    "Bullets": "Wizards",
    "Blackhawks": "Hawks",
    "Braves": "Clippers",
    "SuperSonics": "Thunder",
    "Royals": "Kings",
    "Nationals": "76ers",
    "Bobcats": "Hornets",
}


def color(name: str) -> str:
    """The colour for a team, by current or historical name."""
    return TEAM_COLORS.get(HISTORICAL.get(name, name), FALLBACK)
