# ==============================================================
# PRECOMPUTE ALL TRACK DATA (FINAL VERSION)
# ==============================================================

import pickle
import fastf1
from FastF1_Data_Driven_Simulations import load_data, extract_tire_deg
import os

# Create folder
os.makedirs("tracks", exist_ok=True)

YEAR = 2025

# Load schedule (ALL races)
schedule = fastf1.get_event_schedule(YEAR)

# --------------------------------------------------------------
# TRACK-SPECIFIC PARAMETERS (ALL TRACKS COVERED AUTOMATICALLY)
# --------------------------------------------------------------

TRACK_OVERTAKE = {
    "Monaco Grand Prix": 0.15,
    "Singapore Grand Prix": 0.3,
    "Hungarian Grand Prix": 0.35,
    "Imola Grand Prix": 0.4,
    "Zandvoort Grand Prix": 0.45,
    "Las Vegas Grand Prix": 0.6,
    "Italian Grand Prix": 0.9,
    "Azerbaijan Grand Prix": 0.85,
    "Bahrain Grand Prix": 0.65,
    "British Grand Prix": 0.75,
    "Saudi Arabian Grand Prix": 0.7,
    "Australian Grand Prix": 0.7,
    "Japanese Grand Prix": 0.6,
    "Spanish Grand Prix": 0.6,
    "Canadian Grand Prix": 0.8,
    "Austrian Grand Prix": 0.75,
    "Belgian Grand Prix": 0.8,
    "Dutch Grand Prix": 0.45,
    "United States Grand Prix": 0.75,
    "Mexico City Grand Prix": 0.7,
    "Brazilian Grand Prix": 0.7,
    "Qatar Grand Prix": 0.7,
    "Abu Dhabi Grand Prix": 0.65,
    "Chinese Grand Prix": 0.7
}

# --------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------
test_tracks = [
    "Bahrain Grand Prix"
]
schedule = schedule[schedule["EventName"].isin(test_tracks)]

for _, row in schedule.iterrows():
    # Fix naming differences between FastF1 and our dict
    name_map = {
        "Sao Paulo Grand Prix": "Brazilian Grand Prix",
        "Emilia Romagna Grand Prix": "Imola Grand Prix",
        "Dutch Grand Prix": "Zandvoort Grand Prix"
    }

    race_name = name_map.get(row["EventName"], row["EventName"])

    try:
        print(f"\n=== Processing {race_name} ===", flush=True)

        # 🔥 THIS IS THE KEY FIX
        df, sc, vsc = load_data(track=race_name)
        deg = extract_tire_deg(track=race_name)

        # Track-specific parameters
        track_params = {
            "overtake": TRACK_OVERTAKE.get(race_name, 0.6)
        }

        data = {
            "df": df,
            "sc": sc,
            "vsc": vsc,
            "deg": deg,
            # "laps": row["TotalLaps"],
            "laps": 57,
            "track_params": track_params
        }

        filename = f"tracks/{race_name}.pkl"

        with open(filename, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved {race_name}")

    except Exception as e:
        print(f"Skipped {race_name}: {e}")