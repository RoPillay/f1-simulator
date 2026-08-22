import os
import pickle
import time

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from F1env import F1Env
from track_config import drs_for, laps_for, overtake_for

# ==============================================================
# Team color Mapping
# ==============================================================
TEAM_COLORS = {
    "Red Bull": "#0C03FF",
    "Ferrari": "#DC0000",
    "Mercedes": "#00D2BE",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#BE3EBE",
    "Williams": "#0C7DE7",
    "RB": "#2B4562",
    "Sauber": "#52E252",
    "Haas": "#FFFFFF",
}

TIRE_COLORS = {
    "Soft": "#FF3333",
    "Medium": "#FFD700",
    "Hard": "#FFFFFF",
}


# ==============================================================
# Loading circuit layout
# ==============================================================
@st.cache_data(show_spinner="Loading circuit geometry...")
def load_track_map(track_name):
    fastf1.Cache.enable_cache("cache")

    session = fastf1.get_session(2025, track_name, "R")
    session.load()

    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data().dropna(subset=["X", "Y"])

    return pos["X"].to_numpy(), pos["Y"].to_numpy()


# ==============================================================
# Load Calendar
# ==============================================================
@st.cache_data
def get_calendar():
    # include_testing=False -- test events have no race session
    schedule = fastf1.get_event_schedule(2025, include_testing=False)
    schedule = schedule.sort_values("RoundNumber")

    return schedule["EventName"].tolist()


# ==============================================================
# LOAD PRECOMPUTED TRACK DATA
# ==============================================================
@st.cache_data
def load_track_data(track_name):
    path = f"tracks/{track_name}.pkl"

    if not os.path.exists(path):
        raise FileNotFoundError(track_name)

    with open(path, "rb") as f:
        data = pickle.load(f)

    return (
        data["df"],
        data["sc"],
        data["vsc"],
        data["deg"],
        data.get("laps", laps_for(track_name)),
        data.get("track_params", {"overtake": overtake_for(track_name)}),
    )


# ==============================================================
# SESSION STATE INIT
# ==============================================================
if "env" not in st.session_state:
    st.session_state.env = None
    st.session_state.state = None
    st.session_state.running = False
    st.session_state.player_idx = None
    st.session_state.pit_requested = False
    st.session_state.pit_type = "Medium"
    st.session_state.events = []


# ==============================================================
# UI SETUP
# ==============================================================
st.set_page_config(layout="wide", page_title="F1 Strategy Simulator")

st.title("🏎️ F1 Strategy Simulator")

calendar = get_calendar()

track_name = st.sidebar.selectbox("Select Race", calendar)

# ---- guard: track must be precomputed ----
try:
    df, _, _, _, full_laps, _ = load_track_data(track_name)
except FileNotFoundError:
    st.sidebar.error(f"'{track_name}' has not been precomputed yet.")
    st.warning(
        f"**{track_name}** has no precomputed data.\n\n"
        f"Generate it by running:\n\n"
        f"```\npython precomputing_track_data.py \"{track_name}\"\n```\n\n"
        f"Or precompute the whole calendar with `python precomputing_track_data.py`."
    )
    st.stop()

driver_options = [f"{row['Driver']} ({row['Team']})" for _, row in df.iterrows()]

selected_driver_display = st.sidebar.selectbox("Select Driver", driver_options)
driver_choice = selected_driver_display.split(" ")[0]

start_tire = st.sidebar.selectbox("Starting Tire", ["Soft", "Medium", "Hard"])

# ---- race distance ----
distance_pct = st.sidebar.selectbox(
    "Race Distance",
    ["25%", "50%", "100%"],
    index=0,
    help=f"Full distance at {track_name} is {full_laps} laps",
)

race_laps = max(5, round(full_laps * int(distance_pct.rstrip("%")) / 100))
st.sidebar.caption(f"{race_laps} laps (full distance: {full_laps})")

# ---- sim speed: seconds of wall clock per simulated lap ----
sim_speed = st.sidebar.selectbox("Race Speed", ["Slow", "Normal", "Fast", "Fast Forward"], index=1)

SPEED_MAP = {"Slow": 2.0, "Normal": 1.0, "Fast": 0.4, "Fast Forward": 0.1}
lap_delay = SPEED_MAP[sim_speed]

start_button = st.sidebar.button("🚦 Start Race", type="primary")

if st.session_state.running and st.sidebar.button("⏹️ Stop"):
    st.session_state.running = False
    st.rerun()


# ==============================================================
# START RACE
# ==============================================================
if start_button:

    df, sc, vsc, deg, full_laps, params = load_track_data(track_name)

    env = F1Env(df, sc, vsc, deg, total_laps=race_laps, full_race_laps=full_laps)
    env.track_overtake_factor = params["overtake"]
    env.drs_zones = drs_for(track_name)

    state = env.reset()

    player_idx = list(env.drivers).index(driver_choice)
    env.compounds[player_idx] = start_tire
    state = env._get_state()

    st.session_state.env = env
    st.session_state.state = state
    st.session_state.player_idx = player_idx
    st.session_state.running = True
    st.session_state.pit_requested = False
    st.session_state.events = []

    st.rerun()


# ==============================================================
# NOT RUNNING -> IDLE SCREEN
# ==============================================================
if not st.session_state.running:

    if st.session_state.state is None:
        st.info("Pick a race, driver and starting tire in the sidebar, then hit **Start Race**.")
    else:
        # show the final classification from the last race
        env = st.session_state.env
        state = st.session_state.state

        st.subheader("🏁 Final Classification")

        order = np.argsort(state["gaps"])
        leader = state["gaps"][order[0]]

        rows = []
        for pos, i in enumerate(order):
            retired = not np.isfinite(state["gaps"][i])
            started = int(state["grid"][i])
            moved = started - pos

            rows.append({
                "Pos": "DNF" if retired else pos + 1,
                "Driver": env.drivers[i],
                "Grid": started + 1,
                "+/-": "-" if retired else (f"{moved:+d}" if moved else "="),
                "Gap": "DNF" if retired else ("Leader" if pos == 0 else f"+{state['gaps'][i] - leader:.2f}s"),
                "Tire": state["compound"][i],
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.stop()


# ==============================================================
# ONE LAP PER RERUN
# ==============================================================
env = st.session_state.env
state = st.session_state.state
player_idx = st.session_state.player_idx

# ---------------- player action ----------------
player_action = 0

if st.session_state.pit_requested:
    player_action = {"Soft": 1, "Medium": 2, "Hard": 3}[st.session_state.pit_type]
    st.session_state.pit_requested = False

# ---------------- AI actions ----------------
actions = [0] * len(env.drivers)

for i in range(len(env.drivers)):

    if i == player_idx:
        continue

    age = state["tire_age"][i]
    compound = state["compound"][i]

    if compound == "Soft" and age > np.random.randint(10, 14):
        actions[i] = 2
    elif compound == "Medium" and age > np.random.randint(20, 26):
        actions[i] = 3
    elif compound == "Hard" and age > np.random.randint(32, 40):
        actions[i] = 2

actions[player_idx] = player_action

# ---------------- STEP (one full lap) ----------------
state, reward, done, info = env.step(actions)
st.session_state.state = state

for attacker, defender in info.get("overtakes", []):
    st.session_state.events.append(f"L{state['lap']}  ⚡ {attacker} passes {defender}")

st.session_state.events = st.session_state.events[-8:]


# ==============================================================
# MAIN DISPLAY
# ==============================================================
col_left, col_center, col_right = st.columns([1, 3, 1])

# ---------------- Center circuit view ----------------
with col_center:
    st.subheader("🗺️ Track View")

    try:
        x, y = load_track_map(track_name)
    except Exception as e:
        x, y = None, None
        st.error(f"Could not load circuit geometry: {e}")

    if x is not None:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="#555555", width=6),
            hoverinfo="skip", showlegend=False,
        ))

        # ---------------- DRS ZONES ----------------
        for start, end in drs_for(track_name):
            s_idx = int(start * (len(x) - 1))
            e_idx = int(end * (len(x) - 1))

            fig.add_trace(go.Scatter(
                x=x[s_idx:e_idx], y=y[s_idx:e_idx], mode="lines",
                line=dict(color="lime", width=6),
                opacity=0.4, hoverinfo="skip", showlegend=False,
            ))

        # ---------------- CARS ----------------
        # positions come straight from race time, so the order drawn
        # here always matches the timing screen
        order = np.argsort(state["gaps"])

        for pos, i in enumerate(order):

            if state["retired"][i]:
                continue

            driver = env.drivers[i]
            is_player = i == player_idx

            idx = min(int(state["lap_progress"][i] * (len(x) - 1)), len(x) - 1)

            team = df.loc[df["Driver"] == driver, "Team"].values[0] if "Team" in df.columns else "Unknown"
            color = TEAM_COLORS.get(team, "white")

            gap_ahead = 0.0 if pos == 0 else state["gaps"][i] - state["gaps"][order[pos - 1]]

            fig.add_trace(go.Scatter(
                x=[x[idx]], y=[y[idx]], mode="markers",
                marker=dict(
                    size=18 if is_player else 11,
                    color=color,
                    line=dict(width=3 if is_player else 1,
                              color="yellow" if is_player else "black"),
                ),
                hovertext=(
                    f"<b>{driver}</b><br>Team: {team}<br>"
                    f"P{pos + 1}<br>Tire: {state['compound'][i]} "
                    f"({int(state['tire_age'][i])} laps)<br>"
                    f"Gap ahead: {gap_ahead:.2f}s"
                ),
                hoverinfo="text", showlegend=False,
            ))

        fig.update_layout(
            plot_bgcolor="black", paper_bgcolor="black",
            font=dict(color="white"), showlegend=False, height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False),
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------- Left (Leaderboard) ----------------
with col_left:
    st.subheader("🏁 Leaderboard")

    order = np.argsort(state["gaps"])
    leader_time = state["gaps"][order[0]]

    for pos, i in enumerate(order):
        driver = env.drivers[i]
        marker = "**" if i == player_idx else ""

        if state["retired"][i]:
            st.write(f"{marker}DNF {driver}{marker}")
            continue

        gap = state["gaps"][i] - leader_time
        gap_txt = "Leader" if pos == 0 else f"+{gap:.2f}s"

        st.write(f"{marker}P{pos + 1} {driver}{marker} · {gap_txt} · {state['compound'][i][0]}")

# ---------------- Right (Driver Info) ----------------
with col_right:
    st.subheader("📊 Your Race")

    st.metric("Lap", f"{state['lap']} / {env.total_laps}")

    if state["retired"][player_idx]:
        st.error("💥 You retired")
    else:
        order = np.argsort(state["gaps"])
        pos = int(np.where(order == player_idx)[0][0])

        started = int(state["grid"][player_idx])
        gained = started - pos

        st.metric(
            "Position",
            f"P{pos + 1}",
            delta=(f"{gained:+d}" if gained else None),
        )
        st.caption(f"Started P{started + 1}")

        compound = state["compound"][player_idx]
        age = int(state["tire_age"][player_idx])

        st.markdown(
            f"Tire: <span style='color:{TIRE_COLORS.get(compound, 'white')}'>"
            f"**{compound}**</span> · {age} laps",
            unsafe_allow_html=True,
        )

        if pos > 0:
            st.write(f"⬆ Ahead: {state['gaps'][player_idx] - state['gaps'][order[pos - 1]]:.2f}s")

        if pos < len(order) - 1 and np.isfinite(state["gaps"][order[pos + 1]]):
            st.write(f"⬇ Behind: {state['gaps'][order[pos + 1]] - state['gaps'][player_idx]:.2f}s")


# ==============================================================
# PIT SYSTEM
# ==============================================================
st.subheader("🛠️ Pit Strategy")

colA, colB = st.columns([1, 2])

with colA:
    if st.button("🔧 PIT NEXT LAP", use_container_width=True):
        st.session_state.pit_requested = True

with colB:
    st.session_state.pit_type = st.selectbox(
        "Select Tire", ["Soft", "Medium", "Hard"],
        index=["Soft", "Medium", "Hard"].index(st.session_state.pit_type),
    )

if st.session_state.pit_requested:
    st.success(f"🟢 Box confirmed — {st.session_state.pit_type} on the next lap")


# ==============================================================
# ALERTS
# ==============================================================
if not state["retired"][player_idx]:
    compound = state["compound"][player_idx]
    age = state["tire_age"][player_idx]

    if compound == "Soft" and age > 12:
        st.warning("🔴 Soft tire dropping off — PIT soon")
    elif compound == "Medium" and age > 22:
        st.warning("🟠 Medium tire degrading — consider pit")
    elif compound == "Hard" and age > 32:
        st.warning("🟡 Hard tire aging — strategy window")

if state["safety"] == "SC":
    st.error("🚨 SAFETY CAR DEPLOYED — PIT WINDOW OPEN")
elif state["safety"] == "VSC":
    st.warning("⚠️ VIRTUAL SAFETY CAR — PIT ADVANTAGE")

if st.session_state.events:
    with st.expander("📻 Race events", expanded=True):
        for e in reversed(st.session_state.events):
            st.write(e)


# ==============================================================
# END RACE / AUTO ADVANCE
# ==============================================================
if done:
    st.session_state.running = False
    st.success("🏁 Race Finished!")
    st.rerun()

time.sleep(lap_delay)
st.rerun()
