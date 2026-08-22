# ==============================================================
# SHARED TRACK CONFIGURATION
# ==============================================================
#
# Single source of truth for per-circuit data, imported by BOTH
# precomputing_track_data.py and app.py.
#
# Everything is keyed by norm(), not by raw event name, so that
# "Emilia Romagna Grand Prix", "Imola" and "imola gp" all resolve
# to the same entry and the two sides can never disagree again.
#
# ==============================================================

import unicodedata


# --------------------------------------------------------------
# NAME NORMALISATION
# --------------------------------------------------------------
def norm(name):
    """Collapse an event name to a stable lookup key."""

    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("grand prix", "").replace("gp", "")

    return " ".join(s.split())


# --------------------------------------------------------------
# ALIASES (nickname -> canonical key)
# --------------------------------------------------------------
ALIASES = {
    "imola": "emilia romagna",
    "zandvoort": "dutch",
    "brazilian": "sao paulo",
    "interlagos": "sao paulo",
    "monza": "italian",
    "spa": "belgian",
    "silverstone": "british",
    "suzuka": "japanese",
    "baku": "azerbaijan",
    "jeddah": "saudi arabian",
    "cota": "united states",
    "mexican": "mexico city",
}


def key_for(name):
    """Normalised lookup key for an event name, resolving aliases."""

    k = norm(name)

    return ALIASES.get(k, k)


# --------------------------------------------------------------
# RACE DISTANCE (real scheduled lap count per circuit)
# --------------------------------------------------------------
TRACK_LAPS = {
    "australian": 58,
    "chinese": 56,
    "japanese": 53,
    "bahrain": 57,
    "saudi arabian": 50,
    "miami": 57,
    "emilia romagna": 63,
    "monaco": 78,
    "spanish": 66,
    "canadian": 70,
    "austrian": 71,
    "british": 52,
    "belgian": 44,
    "hungarian": 70,
    "dutch": 72,
    "italian": 53,
    "azerbaijan": 51,
    "singapore": 62,
    "united states": 56,
    "mexico city": 71,
    "sao paulo": 71,
    "las vegas": 50,
    "qatar": 57,
    "abu dhabi": 58,
}

# --------------------------------------------------------------
# TRACK-SPECIFIC OVERTAKING DIFFICULTY
# --------------------------------------------------------------
TRACK_OVERTAKE = {
    "monaco": 0.15,
    "singapore": 0.3,
    "hungarian": 0.35,
    "emilia romagna": 0.4,
    "dutch": 0.45,
    "las vegas": 0.6,
    "italian": 0.9,
    "azerbaijan": 0.85,
    "bahrain": 0.65,
    "british": 0.75,
    "saudi arabian": 0.7,
    "australian": 0.7,
    "japanese": 0.6,
    "spanish": 0.6,
    "canadian": 0.8,
    "austrian": 0.75,
    "belgian": 0.8,
    "united states": 0.75,
    "mexico city": 0.7,
    "sao paulo": 0.7,
    "qatar": 0.7,
    "abu dhabi": 0.65,
    "chinese": 0.7,
}

# --------------------------------------------------------------
# DRS DETECTION ZONES (as fractions of a lap)
#
# NOTE: these are hand-estimated, not surveyed positions. They are
# good enough to weight overtaking, but should be replaced with real
# zone geometry when the canvas renderer lands.
# --------------------------------------------------------------
DRS_ZONES = {
    "australian": [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.85)],
    "chinese": [(0.2, 0.35), (0.7, 0.9)],
    "japanese": [(0.85, 0.98)],
    "bahrain": [(0.1, 0.25), (0.45, 0.6), (0.75, 0.95)],
    "saudi arabian": [(0.2, 0.35), (0.5, 0.65), (0.75, 0.9)],
    "miami": [(0.1, 0.25), (0.4, 0.6), (0.7, 0.85)],
    "emilia romagna": [(0.8, 0.95)],
    "monaco": [(0.85, 0.98)],
    "spanish": [(0.6, 0.75), (0.1, 0.25)],
    "canadian": [(0.2, 0.35), (0.5, 0.7), (0.8, 0.95)],
    "austrian": [(0.1, 0.25), (0.3, 0.5), (0.6, 0.8)],
    "british": [(0.3, 0.5), (0.6, 0.8)],
    "belgian": [(0.1, 0.3), (0.7, 0.95)],
    "hungarian": [(0.6, 0.75), (0.8, 0.95)],
    "dutch": [(0.6, 0.75), (0.8, 0.9)],
    "italian": [(0.1, 0.3), (0.6, 0.8)],
    "azerbaijan": [(0.1, 0.3), (0.6, 0.95)],
    "singapore": [(0.2, 0.35), (0.45, 0.6), (0.7, 0.85)],
    "united states": [(0.3, 0.5), (0.7, 0.9)],
    "mexico city": [(0.2, 0.35), (0.4, 0.6), (0.7, 0.9)],
    "sao paulo": [(0.4, 0.6), (0.7, 0.9)],
    "las vegas": [(0.2, 0.4), (0.7, 0.9)],
    "qatar": [(0.8, 0.98)],
    "abu dhabi": [(0.2, 0.4), (0.7, 0.9)],
}

DEFAULT_LAPS = 57
DEFAULT_OVERTAKE = 0.6


# --------------------------------------------------------------
# ACCESSORS
# --------------------------------------------------------------
def laps_for(name):
    return TRACK_LAPS.get(key_for(name), DEFAULT_LAPS)


def overtake_for(name):
    return TRACK_OVERTAKE.get(key_for(name), DEFAULT_OVERTAKE)


def drs_for(name):
    return DRS_ZONES.get(key_for(name), [])
