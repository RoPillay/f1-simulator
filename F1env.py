# ==============================================================
# F1 ENVIRONMENT (STEP-BASED SIMULATOR FOR UI + RL)
# ==============================================================

import numpy as np
import random

# ==============================================================
# ENVIRONMENT CLASS
# ==============================================================

class F1Env:

    def __init__(self, df, sc_prob, vsc_prob, deg_model):

        # -----------------------------
        # DATA
        # -----------------------------
        self.df = df
        self.sc_prob = sc_prob
        self.vsc_prob = vsc_prob
        self.deg_model = deg_model
        self.drs_zones = []

        self.drivers = df["Driver"].values
        self.base = df["Baseline"].values * 1.02
        self.var = df["LapVar"].clip(0.15, 0.45).values
        self.dnf = df["DNFProb"].values

        self.n = len(self.drivers)

        # -----------------------------
        # TIRE MODEL (same structure as your sim)
        # -----------------------------
        self.tire = {
            "Soft": {
                "offset": -0.6,
                "deg": deg_model["SOFT"] * 1.3,
                "curve": 0.003
            },
            "Medium": {
                "offset": 0.0,
                "deg": deg_model["MEDIUM"] * 1.1,
                "curve": 0.002
            },
            "Hard": {
                "offset": 0.4,
                "deg": deg_model["HARD"] * 0.9,
                "curve": 0.0001
            }
        }

        # -----------------------------
        # PARAMETERS
        # -----------------------------
        self.pit_mean = 23.5
        self.pit_sd = 1.5

        self.traffic_penalty = (0.15, 0.35)
        self.drs_range = 1.0
        self.overtake_distance = 2.0
        self.track_overtake_factor = 0.65
        self.failed_overtake_penalty = 0.1
        self.track_temp = 35  # Bahrain typical (°C)

    # ==========================================================
    # RESET (START NEW RACE)
    # ==========================================================
    def reset(self):

        self.lap = 0

        self.safety_active = None
        self.safety_timer = 0

        self.total_time = np.zeros(self.n)
        self.position = np.arange(self.n)

        # initial compounds (all medium to start)
        self.compounds = ["Medium"] * self.n

        # tire tracking
        self.stint_laps = np.zeros(self.n)

        # freshness bonus (like your sim)
        self.fresh = np.zeros(self.n)

        # -----------------------------
        # LAP PROGRESS (0 → 1)
        # -----------------------------
        self.lap_progress = np.zeros(self.n)

        # -----------------------------
        # DRIVER STYLE (push vs save)
        # -----------------------------
        self.driving_style = np.random.choice(
            ["push", "normal", "save"],
            size=self.n,
            p=[0.3, 0.5, 0.2]
        )

        # initialize grid using base pace
        order = np.argsort(self.base)
        self.position[order] = np.arange(self.n)

        return self._get_state()

    # ==========================================================
    # STEP (ONE LAP)
    # ==========================================================
    def step(self, actions):
        """
        actions:
        0 = stay out
        1 = pit soft
        2 = pit medium
        3 = pit hard
        """
        
        lap_times = np.zeros(self.n)
        overtake_events = []

        # -----------------------------
        # DNF CHECK
        # -----------------------------
        for i in range(self.n):
            if np.random.rand() < self.dnf[i] / 57:
                self.total_time[i] = np.inf

        # -----------------------------
        # SAFETY CAR / VSC
        # -----------------------------
       # SAFETY CAR LOGIC
        if self.safety_timer > 0:
            safety = self.safety_active
            self.safety_timer -= 1
        else:
            safety = None

            if np.random.rand() < self.sc_prob:
                self.safety_active = "SC"
                self.safety_timer = np.random.randint(3, 6)
                safety = "SC"

            elif np.random.rand() < self.vsc_prob:
                self.safety_active = "VSC"
                self.safety_timer = np.random.randint(2, 4)
                safety = "VSC"

        # -----------------------------
        # APPLY ACTIONS (PITS)
        # -----------------------------
        for i in range(self.n):

            if not np.isfinite(self.total_time[i]):
                continue

            if actions[i] == 1:
                self.compounds[i] = "Soft"
                self.stint_laps[i] = 0
                self.total_time[i] += np.random.normal(self.pit_mean, self.pit_sd)
                self.fresh[i] = 3

            elif actions[i] == 2:
                self.compounds[i] = "Medium"
                self.stint_laps[i] = 0
                self.total_time[i] += np.random.normal(self.pit_mean, self.pit_sd)
                self.fresh[i] = 3

            elif actions[i] == 3:
                self.compounds[i] = "Hard"
                self.stint_laps[i] = 0
                self.total_time[i] += np.random.normal(self.pit_mean, self.pit_sd)
                self.fresh[i] = 3

        # -----------------------------
        # LAP SIMULATION
        # -----------------------------
        for i in range(self.n):

            if not np.isfinite(self.total_time[i]):
                continue

            t = self.tire[self.compounds[i]]

            age = self.stint_laps[i]
            compound = self.compounds[i]

            # -----------------------------
            # BASE DEGRADATION (from real data)
            # -----------------------------
            base_deg = t["deg"]

            # -----------------------------
            # TRACK TEMPERATURE EFFECT
            # -----------------------------
            temp_factor = (self.track_temp - 30) / 20  # normalized

            deg_effect = base_deg * age * (1 + 0.3 * temp_factor)

            # quadratic growth (natural wear)
            deg_effect += 0.0015 * (age ** 2)

            # -----------------------------
            # COMPOUND-SPECIFIC BEHAVIOR
            # -----------------------------
            if compound == "Soft":
                if age > 10:
                    deg_effect += 0.08 * (age - 10)
                if age > 15:
                    deg_effect += 0.15 * (age - 15)

            elif compound == "Medium":
                if age > 18:
                    deg_effect += 0.06 * (age - 18)
                if age > 28:
                    deg_effect += 0.12 * (age - 28)

            elif compound == "Hard":
                if age > 25:
                    deg_effect += 0.04 * (age - 25)
                if age > 40:
                    deg_effect += 0.08 * (age - 40)

            # -----------------------------
            # DRIVER STYLE EFFECT
            # -----------------------------
            style = self.driving_style[i]

            if style == "push":
                deg_effect *= 1.25
                style_offset = -0.2
            elif style == "save":
                deg_effect *= 0.75
                style_offset = +0.2
            else:
                style_offset = 0.0

            # -----------------------------
            # FINAL LAP TIME
            # -----------------------------
            lap_time = (
                self.base[i]
                + t["offset"]
                + deg_effect
                + style_offset
                + np.random.normal(0, self.var[i] * 0.35)
            )

            # fresh tire bonus
            if self.fresh[i] > 0:
                lap_time -= 0.4
                self.fresh[i] -= 1

            # safety car override
            if safety == "SC":
                lap_time = 130
            elif safety == "VSC":
                lap_time = 115

            lap_times[i] = lap_time
            self.stint_laps[i] += 1

        # -----------------------------
        # FIX 5: CAP EXTREME LAP TIMES
        # -----------------------------
        lap_times = np.clip(
            lap_times,
            self.base - 1.5,
            self.base + 2.0
        )    

        # -----------------------------
        # APPLY LAP TIMES
        # -----------------------------
        self.total_time += lap_times

        # -----------------------------
        # PACE COMPRESSION 
        # -----------------------------
        valid_times = self.total_time[np.isfinite(self.total_time)]

        if len(valid_times) > 0:
            leader_time = np.min(valid_times)

            for i in range(self.n):
                if np.isfinite(self.total_time[i]):
                    gap = self.total_time[i] - leader_time
                    self.total_time[i] = leader_time + gap * 0.995

        # -----------------------------
        # TRAFFIC EFFECTS
        # -----------------------------
        order = np.argsort(self.total_time)

        for p in range(1, self.n):
            d = order[p]
            a = order[p - 1]

            if not np.isfinite(self.total_time[d]) or not np.isfinite(self.total_time[a]):
                continue

            if self.total_time[d] - self.total_time[a] < 0.7:
                self.total_time[d] += np.random.uniform(*self.traffic_penalty)

        # -----------------------------
        # OVERTAKES
        # -----------------------------
        for p in range(1, self.n):

            d = order[p]
            a = order[p - 1]

            if not np.isfinite(self.total_time[d]) or not np.isfinite(self.total_time[a]):
                continue

            gap = self.total_time[d] - self.total_time[a]

            if gap > self.overtake_distance:
                continue

            # Check if in DRS zone
            in_drs_zone = False

            for start, end in self.drs_zones:
                if start <= self.lap_progress[d] <= end:
                    in_drs_zone = True
                    break

            drs = 1 if (gap < self.drs_range and in_drs_zone) else 0
            delta = self.base[a] - self.base[d]

            prob = 1 / (1 + np.exp(-(-1.2 + 2.5 * delta + 2.5 * drs)))

            if np.random.rand() < self.track_overtake_factor:

                if np.random.rand() < prob:

                    # ============================
                    # CONTINUOUS OVERTAKE BOOST
                    # ============================
                    progress_boost = np.random.uniform(0.01, 0.03)

                    self.lap_progress[d] += progress_boost

                    # If passes ahead car → complete overtake
                    if self.lap_progress[d] > self.lap_progress[a]:

                        gain = np.random.uniform(0.3, 0.8)

                        self.total_time[d] -= gain
                        self.total_time[a] += gain * 0.3

                        overtake_events.append((self.drivers[d], self.drivers[a]))

                else:
                    # FAILED OVERTAKE
                    self.total_time[d] += np.random.uniform(0.1, 0.3)

        # -----------------------------
        # UPDATE POSITIONS
        # -----------------------------
        order = np.argsort(self.total_time)
        self.position[order] = np.arange(self.n)

        # only increment lap when full lap completed
        completed = self.lap_progress >= 1.0

        self.lap += int(np.any(completed))
        self.lap_progress[completed] -= 1.0

        # -----------------------------
        # DONE
        # -----------------------------
        done = self.lap >= getattr(self, "total_laps", 57)

        # -----------------------------
        # REWARD (can change later)
        # -----------------------------
        reward = -self.position

        # -----------------------------
        # UPDATE LAP PROGRESS
        # -----------------------------

        # progress increment based on pace
        speed_factor = 1 / (lap_times + 1e-6)

        # normalize speeds
        speed_norm = speed_factor / np.max(speed_factor)

        # update progress
        self.lap_progress += speed_norm * 0.005

        # wrap around lap
        self.lap_progress = self.lap_progress % 1.0

        return self._get_state(), reward, done, {"overtakes": overtake_events}

    # ==========================================================
    # STATE
    # ==========================================================
    def _get_state(self):

        return {
            "lap": self.lap,
            "safety": self.safety_active,
            "position": self.position.copy(),
            "tire_age": self.stint_laps.copy(),
            "compound": self.compounds.copy(),
            "gaps": self.total_time.copy(),
            "lap_progress": self.lap_progress.copy()
        }


# ==============================================================
# SIMPLE TEST (RUN THIS FILE DIRECTLY)
# ==============================================================

if __name__ == "__main__":

    print("Testing F1Env...")

    # IMPORT YOUR EXISTING FUNCTIONS
    from FastF1_Data_Driven_Simulations import load_data, extract_tire_deg

    df, sc, vsc = load_data()
    deg = extract_tire_deg()

    env = F1Env(df, sc, vsc, deg)

    state = env.reset()

    player_driver = "VER"  # change later dynamically
    player_idx = list(env.drivers).index(player_driver)

    # -----------------------------
    # CHOOSE STARTING TIRE
    # -----------------------------
    start_tire = input("Choose starting tire (soft / medium / hard): ").lower()

    if start_tire == "soft":
        env.compounds[player_idx] = "Soft"
    elif start_tire == "hard":
        env.compounds[player_idx] = "Hard"
    else:
        env.compounds[player_idx] = "Medium"

    done = False

    while not done:

        actions = [0] * len(df)

        for i in range(len(df)):

            if i == player_idx:
                continue

            age = state["tire_age"][i]
            compound = state["compound"][i]

            # random thresholds (THIS FIXES SYNCHRONIZATION)
            soft_thresh = np.random.randint(10, 14)
            medium_thresh = np.random.randint(20, 26)
            hard_thresh = np.random.randint(32, 40)

            if compound == "Soft" and age > soft_thresh:
                actions[i] = 2  # soft → medium

            elif compound == "Medium" and age > medium_thresh:
                actions[i] = 3  # medium → hard

            elif compound == "Hard" and age > hard_thresh:
                actions[i] = 2  # hard → medium

        user_input = input("Pit this lap? (0=no, 1=soft, 2=medium, 3=hard): ")
        actions[player_idx] = int(user_input)

        state, reward, done, _ = env.step(actions)

        # -----------------------------
        # 🚨 DNF CHECK (ADD HERE)
        # -----------------------------
        if not np.isfinite(state["gaps"][player_idx]):
            print("\n💥 You DNF’d! Race over.")
            break

        # ==============================
        # PRINT GAME INFO (ADD HERE)
        # ==============================
        print(f"\n--- LAP {state['lap']} ---")

        print(f"Driver: {player_driver}")
        print(f"Position: {state['position'][player_idx] + 1}")
        print(f"Tire: {state['compound'][player_idx]}")
        print(f"Tire Age: {state['tire_age'][player_idx]:.0f}")

        sorted_idx = np.argsort(state["gaps"])
        pos = np.where(sorted_idx == player_idx)[0][0]

        if pos > 0:
            ahead = sorted_idx[pos - 1]
            gap_ahead = state["gaps"][player_idx] - state["gaps"][ahead]
            print(f"Gap Ahead: {gap_ahead:.2f}s")

        if pos < len(sorted_idx) - 1:
            behind = sorted_idx[pos + 1]
            gap_behind = state["gaps"][behind] - state["gaps"][player_idx]
            print(f"Gap Behind: {gap_behind:.2f}s")

    print("Race completed successfully")

    # -----------------------------
    # FINAL CLASSIFICATION
    # -----------------------------
    final_order = np.argsort(state["gaps"])

    leader_time = state["gaps"][final_order[0]]

    for pos, idx in enumerate(final_order):
        driver = env.drivers[idx]

        if np.isfinite(state["gaps"][idx]):

            if pos == 0:
                print(f"P1: {driver} (Winner)")
            else:
                gap = state["gaps"][idx] - leader_time
                print(f"P{pos+1}: {driver} (+{gap:.2f}s)")

        else:
            print(f"P{pos+1}: {driver} (DNF)")