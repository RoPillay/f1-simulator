# ==============================================================
# F1 ENVIRONMENT (LAP-BASED SIMULATOR FOR UI + RL)
# ==============================================================
#
# ONE call to step() == ONE racing lap.
#
# Track position is NOT tracked separately -- it is derived from
# each car's accumulated race time (see positions_at). That keeps
# the cars drawn on the circuit in exactly the same order as the
# timing screen, and makes real time gaps show up as real spacing.
#
# ==============================================================

import bisect
import numpy as np


# ==============================================================
# ENVIRONMENT CLASS
# ==============================================================

class F1Env:

    def __init__(self, df, sc_prob, vsc_prob, deg_model, total_laps=57,
                 full_race_laps=None):

        # -----------------------------
        # DATA
        # -----------------------------
        self.df = df

        # sc_prob / vsc_prob arrive as PER-RACE figures (the fraction of
        # races that saw a Safety Car). They are converted to a per-lap
        # deployment chance in step() -- rolling a per-race probability
        # once per lap is what pinned the Safety Car on permanently.
        self.sc_prob = sc_prob
        self.vsc_prob = vsc_prob

        self.deg_model = deg_model
        self.drs_zones = []

        self.drivers = df["Driver"].values
        self.base = df["Baseline"].values * 1.02
        self.var = df["LapVar"].clip(0.15, 0.45).values
        self.dnf = df["DNFProb"].values

        self.n = len(self.drivers)
        self.total_laps = total_laps

        # The rate at which Safety Cars appear is a property of the
        # circuit, not of how long you chose to race. Dividing by the
        # FULL scheduled distance keeps that rate constant, so a 25%
        # race sees roughly a quarter as many interventions.
        self.full_race_laps = full_race_laps or total_laps

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
        self.track_temp = 35  # Bahrain typical (deg C)

        # -----------------------------
        # SAFETY CAR LAP TIMES
        # -----------------------------
        self.sc_lap_time = 130.0
        self.vsc_lap_time = 115.0

        # -----------------------------
        # SANITY GUARD ONLY (seconds around base pace)
        #
        # This is NOT a degradation cap -- degradation and the tire
        # cliff are allowed to express fully. It exists purely to
        # stop a numerical blow-up producing a nonsense lap time.
        # -----------------------------
        self.lap_time_floor = -2.0
        self.lap_time_ceiling = 15.0

    # ==========================================================
    # RESET (START NEW RACE)
    # ==========================================================
    def reset(self):

        self.lap = 0

        self.safety_active = None
        self.safety_timer = 0
        self.last_safety = None

        self.total_time = np.zeros(self.n)
        self.position = np.arange(self.n)

        # initial compounds (all medium to start)
        self.compounds = ["Medium"] * self.n

        # tire tracking
        self.stint_laps = np.zeros(self.n)

        # freshness bonus (like your sim)
        self.fresh = np.zeros(self.n)

        # -----------------------------
        # LAP COMPLETION HISTORY
        #
        # completion_times[i][k] = race time at which car i crossed
        # the line to complete lap k+1. This is what positions_at
        # interpolates between to place cars on the circuit.
        # -----------------------------
        self.completion_times = [[] for _ in range(self.n)]

        # laps completed at the moment a car retired (None = running)
        self.retired_at = [None] * self.n

        # race clock used for rendering (leader's elapsed time)
        self.race_clock = 0.0

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
    # DRS COVERAGE
    # ==========================================================
    def _drs_coverage(self):
        """Fraction of the lap that sits inside a DRS zone."""

        if not self.drs_zones:
            return 0.0

        total = sum(max(0.0, end - start) for start, end in self.drs_zones)

        return float(np.clip(total, 0.0, 1.0))

    # ==========================================================
    # STEP (ONE FULL LAP)
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
        # SAFETY CAR / VSC
        # -----------------------------
        ref_laps = max(self.full_race_laps, 1)

        sc_lap_prob = float(np.clip(self.sc_prob / ref_laps, 0.0, 1.0))
        vsc_lap_prob = float(np.clip(self.vsc_prob / ref_laps, 0.0, 1.0))

        if self.safety_timer > 0:
            safety = self.safety_active
            self.safety_timer -= 1

            if self.safety_timer == 0:
                self.safety_active = None
        else:
            safety = None

            if np.random.rand() < sc_lap_prob:
                self.safety_active = "SC"
                self.safety_timer = np.random.randint(3, 6)
                safety = "SC"

            elif np.random.rand() < vsc_lap_prob:
                self.safety_active = "VSC"
                self.safety_timer = np.random.randint(2, 4)
                safety = "VSC"

        # what actually governed THIS lap (safety_active may already
        # have been cleared above on the lap the period expires)
        self.last_safety = safety

        # -----------------------------
        # DNF CHECK (ONCE PER LAP)
        #
        # dnf[i] is a whole-race probability, so convert it to the
        # equivalent per-lap hazard rather than rolling it raw.
        # -----------------------------
        for i in range(self.n):

            if not np.isfinite(self.total_time[i]):
                continue

            race_p = float(np.clip(self.dnf[i], 0.0, 0.99))
            lap_p = 1.0 - (1.0 - race_p) ** (1.0 / max(self.total_laps, 1))

            if np.random.rand() < lap_p:
                self.total_time[i] = np.inf
                self.retired_at[i] = len(self.completion_times[i])

        # -----------------------------
        # APPLY ACTIONS (PITS)
        # -----------------------------
        pit_loss = np.zeros(self.n)

        for i in range(self.n):

            if not np.isfinite(self.total_time[i]):
                continue

            if actions[i] in (1, 2, 3):

                self.compounds[i] = {1: "Soft", 2: "Medium", 3: "Hard"}[actions[i]]
                self.stint_laps[i] = 0
                self.fresh[i] = 3

                pit_loss[i] = np.random.normal(self.pit_mean, self.pit_sd)

        # -----------------------------
        # LAP SIMULATION (GREEN FLAG PACE)
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
            # COMPOUND-SPECIFIC BEHAVIOR (THE CLIFF)
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

            # sanity guard only -- the cliff above is left intact
            lap_time = float(np.clip(
                lap_time,
                self.base[i] + self.lap_time_floor,
                self.base[i] + self.lap_time_ceiling
            ))

            lap_times[i] = lap_time
            self.stint_laps[i] += 1

        # -----------------------------
        # SAFETY CAR OVERRIDE
        #
        # Applied AFTER the sanity guard so a Safety Car lap is
        # actually slow instead of being clipped back to green pace.
        # -----------------------------
        running = np.isfinite(self.total_time)

        if safety == "SC":
            lap_times[running] = self.sc_lap_time
        elif safety == "VSC":
            lap_times[running] = self.vsc_lap_time

        # -----------------------------
        # APPLY LAP TIMES + PIT LOSS
        # -----------------------------
        self.total_time += lap_times + pit_loss

        # -----------------------------
        # TRAFFIC EFFECTS (DIRTY AIR)
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
        drs_coverage = self._drs_coverage()

        order = np.argsort(self.total_time)

        for p in range(1, self.n):

            d = order[p]
            a = order[p - 1]

            if not np.isfinite(self.total_time[d]) or not np.isfinite(self.total_time[a]):
                continue

            # no overtaking under Safety Car
            if safety in ("SC", "VSC"):
                continue

            gap = self.total_time[d] - self.total_time[a]

            if gap > self.overtake_distance:
                continue

            # DRS is available if close enough AND the move happens
            # to come in a DRS zone (weighted by how much of the lap
            # this circuit's zones actually cover)
            in_drs_zone = np.random.rand() < drs_coverage

            drs = 1 if (gap < self.drs_range and in_drs_zone) else 0

            delta = self.base[a] - self.base[d]

            prob = 1 / (1 + np.exp(-(-1.2 + 2.5 * delta + 2.5 * drs)))

            if np.random.rand() < self.track_overtake_factor:

                if np.random.rand() < prob:

                    # completed pass -- the attacker clears the car ahead
                    gain = gap + np.random.uniform(0.05, 0.25)

                    self.total_time[d] -= gain
                    self.total_time[a] += np.random.uniform(0.05, 0.15)

                    overtake_events.append((self.drivers[d], self.drivers[a]))

                else:
                    # FAILED OVERTAKE
                    self.total_time[d] += np.random.uniform(0.1, 0.3)

        # -----------------------------
        # RECORD LAP COMPLETION
        # -----------------------------
        for i in range(self.n):
            if np.isfinite(self.total_time[i]):
                self.completion_times[i].append(float(self.total_time[i]))

        # -----------------------------
        # UPDATE POSITIONS
        # -----------------------------
        order = np.argsort(self.total_time)
        self.position[order] = np.arange(self.n)

        # -----------------------------
        # ADVANCE LAP COUNTER
        # -----------------------------
        self.lap += 1

        finite = self.total_time[np.isfinite(self.total_time)]
        self.race_clock = float(np.min(finite)) if len(finite) else self.race_clock

        # -----------------------------
        # DONE
        # -----------------------------
        done = (self.lap >= self.total_laps) or (len(finite) == 0)

        # -----------------------------
        # REWARD (can change later)
        # -----------------------------
        reward = -self.position

        return self._get_state(), reward, done, {"overtakes": overtake_events}

    # ==========================================================
    # RENDERING: WHERE IS EACH CAR AT RACE TIME t ?
    # ==========================================================
    def positions_at(self, t=None):
        """
        Distance covered by each car, measured in laps, at race time t.

        Returns a float array: 12.4 means "four tenths of the way
        round lap 13". Sorting by this descending gives exactly the
        running order, so the circuit view and the timing screen can
        never disagree.
        """

        if t is None:
            t = self.race_clock

        dist = np.zeros(self.n)

        for i in range(self.n):

            ct = self.completion_times[i]

            # retired -- freeze the car where it stopped
            if self.retired_at[i] is not None:
                dist[i] = float(self.retired_at[i])
                continue

            if not ct:
                dist[i] = 0.0
                continue

            # laps fully completed by time t
            k = bisect.bisect_right(ct, t)

            if k == 0:
                lap_start = 0.0
                lap_dur = ct[0]
            else:
                lap_start = ct[k - 1]

                if k < len(ct):
                    lap_dur = ct[k] - ct[k - 1]
                elif len(ct) > 1:
                    lap_dur = ct[-1] - ct[-2]
                else:
                    lap_dur = ct[0]

            if lap_dur <= 0:
                frac = 0.0
            else:
                frac = (t - lap_start) / lap_dur

            dist[i] = k + float(np.clip(frac, 0.0, 1.0))

        return dist

    def track_positions(self, t=None):
        """Position around the circuit as a 0-1 fraction, for drawing."""

        return self.positions_at(t) % 1.0

    # ==========================================================
    # STATE
    # ==========================================================
    def _get_state(self):

        distance = self.positions_at()

        return {
            "lap": self.lap,
            "total_laps": self.total_laps,
            "safety": self.last_safety,
            "position": self.position.copy(),
            "tire_age": self.stint_laps.copy(),
            "compound": self.compounds.copy(),
            "gaps": self.total_time.copy(),
            "distance": distance,
            "lap_progress": distance % 1.0,
            "race_clock": self.race_clock,
            "retired": [r is not None for r in self.retired_at],
        }
