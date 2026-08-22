# ==============================================================
# PRECOMPUTE ALL TRACK DATA
# ==============================================================
#
# Writes one tracks/<EventName>.pkl per race.
#
# IMPORTANT: files are saved under FastF1's OWN EventName, which is
# exactly what the app's dropdown shows. Per-circuit parameters are
# looked up through track_config.key_for() instead, so nicknames
# resolve correctly without the filename ever drifting from the name
# the app asks for.
#
# Usage:
#   python precomputing_track_data.py                  # all races
#   python precomputing_track_data.py Bahrain Monaco   # just these
#
# ==============================================================

import os
import pickle
import sys

import fastf1

from FastF1_Data_Driven_Simulations import load_data, extract_tire_deg
from track_config import key_for, laps_for, overtake_for, TRACK_LAPS

# Create folder
os.makedirs("tracks", exist_ok=True)

YEAR = 2025


# --------------------------------------------------------------
# BUILD THE RACE LIST
#
# include_testing=False matters: pre-season test events carry no
# race session and would fail every time.
# --------------------------------------------------------------
schedule = fastf1.get_event_schedule(YEAR, include_testing=False)

# optional CLI filter: match on any substring of the event name
wanted = [a.lower() for a in sys.argv[1:]]

if wanted:
    schedule = schedule[
        schedule["EventName"].str.lower().apply(
            lambda n: any(w in n for w in wanted)
        )
    ]
    print(f"Filtering to {len(schedule)} race(s) matching {wanted}")

print(f"Precomputing {len(schedule)} race(s) for {YEAR}\n")

ok, failed = [], []

# --------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------
for _, row in schedule.iterrows():

    # save under FastF1's own name -- this is what the app looks up
    event_name = row["EventName"]

    try:
        print(f"=== Processing {event_name} ===", flush=True)

        df, sc, vsc = load_data(track=event_name)
        deg = extract_tire_deg(track=event_name)

        laps = laps_for(event_name)

        if key_for(event_name) not in TRACK_LAPS:
            print(f"  ! no lap count known for '{key_for(event_name)}', using {laps}")

        track_params = {
            "overtake": overtake_for(event_name),
        }

        data = {
            "df": df,
            "sc": sc,
            "vsc": vsc,
            "deg": deg,
            "laps": laps,
            "track_params": track_params,
        }

        with open(f"tracks/{event_name}.pkl", "wb") as f:
            pickle.dump(data, f)

        print(f"  saved  ({laps} laps, overtake {track_params['overtake']})")
        ok.append(event_name)

    except Exception as e:
        print(f"  SKIPPED: {type(e).__name__}: {e}")
        failed.append((event_name, f"{type(e).__name__}: {e}"))

# --------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------
print("\n" + "=" * 60)
print(f"DONE: {len(ok)} succeeded, {len(failed)} failed")
print("=" * 60)

if failed:
    print("\nFailed races:")
    for name, err in failed:
        print(f"  {name}: {err}")
