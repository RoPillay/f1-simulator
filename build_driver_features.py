# ==============================================================
# BUILD GLOBAL DRIVER / TEAM FEATURES  (RUN ONCE)
# ==============================================================
#
# Team strength, driver form, qualifying score, race performance and
# DNF rates are properties of a DRIVER and a SEASON -- not of a
# circuit. They are identical for Bahrain and for Monaco.
#
# Each one walks three seasons of race sessions, so computing them
# inside load_data() meant repeating ~300 session loads for every
# track precomputed. That is why they ended up commented out.
#
# This script computes them ONCE and caches the result. load_data()
# then just reads the cache.
#
# Usage:
#   python build_driver_features.py            # build if missing
#   python build_driver_features.py --force    # always rebuild
#
# ==============================================================

import os
import pickle
import sys
import time

import fastf1

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

FEATURES_DIR = "features"
FEATURES_PATH = os.path.join(FEATURES_DIR, "driver_features.pkl")


# --------------------------------------------------------------
# LOADER USED BY load_data()
# --------------------------------------------------------------
def load_features():
    """
    Return the cached feature maps, or None if they have not been
    built yet. Never computes -- building is an explicit choice
    because it downloads three seasons of session data.
    """

    if not os.path.exists(FEATURES_PATH):
        return None

    with open(FEATURES_PATH, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------
# BUILD
# --------------------------------------------------------------
def build():

    # imported here so `load_features` stays cheap to import
    from FastF1_Data_Driven_Simulations import (
        compute_team_strength,
        compute_driver_form,
        compute_quali_score,
        compute_race_performance,
        compute_dnf_rates,
    )

    os.makedirs(FEATURES_DIR, exist_ok=True)

    steps = [
        ("team_strength", compute_team_strength),
        ("driver_form", compute_driver_form),
        ("quali_score", compute_quali_score),
        ("race_perf", compute_race_performance),
        ("dnf_rates", compute_dnf_rates),
    ]

    features = {}

    print("=" * 62)
    print("Building global driver/team features from 2023-2025")
    print("This downloads a lot of session data and is SLOW the")
    print("first time. Subsequent runs reuse the FastF1 cache.")
    print("=" * 62)

    for name, fn in steps:
        print(f"\n--- {name} ---", flush=True)
        t0 = time.time()

        try:
            features[name] = fn()
            print(f"    {len(features[name])} entries in {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            features[name] = {}

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)

    print("\n" + "=" * 62)
    print(f"Saved {FEATURES_PATH}")

    for name in features:
        print(f"  {name:<16} {len(features[name]):>4} entries")

    print("=" * 62)

    return features


if __name__ == "__main__":

    force = "--force" in sys.argv

    if not force and load_features() is not None:
        print(f"{FEATURES_PATH} already exists. Use --force to rebuild.")
    else:
        build()
