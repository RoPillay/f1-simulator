import streamlit as st
import fastf1
import pickle
import numpy as np
import pandas as pd
import time
import os
import plotly.graph_objects as go
from F1env import F1Env

#==============================================================
# Team color Mapping
#==============================================================
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
    "Haas": "#FFFFFF"
}

#==============================================================
# DRS Detection Zones
#==============================================================
DRS_ZONES = {
    "Australian Grand Prix": [(0.1,0.2),(0.3,0.4),(0.5,0.6),(0.7,0.85)],
    "Chinese Grand Prix": [(0.2,0.35),(0.7,0.9)],
    "Japanese Grand Prix": [(0.85,0.98)],
    "Bahrain Grand Prix": [(0.1,0.25),(0.45,0.6),(0.75,0.95)],
    "Saudi Arabian Grand Prix": [(0.2,0.35),(0.5,0.65),(0.75,0.9)],
    "Miami Grand Prix": [(0.1,0.25),(0.4,0.6),(0.7,0.85)],
    "Emilia Romagna Grand Prix": [(0.8,0.95)],
    "Monaco Grand Prix": [(0.85,0.98)],
    "Spanish Grand Prix": [(0.6,0.75),(0.1,0.25)],
    "Canadian Grand Prix": [(0.2,0.35),(0.5,0.7),(0.8,0.95)],
    "Austrian Grand Prix": [(0.1,0.25),(0.3,0.5),(0.6,0.8)],
    "British Grand Prix": [(0.3,0.5),(0.6,0.8)],
    "Belgian Grand Prix": [(0.1,0.3),(0.7,0.95)],
    "Hungarian Grand Prix": [(0.6,0.75),(0.8,0.95)],
    "Dutch Grand Prix": [(0.6,0.75),(0.8,0.9)],
    "Italian Grand Prix": [(0.1,0.3),(0.6,0.8)],
    "Azerbaijan Grand Prix": [(0.1,0.3),(0.6,0.95)],
    "Singapore Grand Prix": [(0.2,0.35),(0.45,0.6),(0.7,0.85)],
    "United States Grand Prix": [(0.3,0.5),(0.7,0.9)],
    "Mexico City Grand Prix": [(0.2,0.35),(0.4,0.6),(0.7,0.9)],
    "Brazilian Grand Prix": [(0.4,0.6),(0.7,0.9)],
    "Las Vegas Grand Prix": [(0.2,0.4),(0.7,0.9)],
    "Qatar Grand Prix": [(0.8,0.98)],
    "Abu Dhabi Grand Prix": [(0.2,0.4),(0.7,0.9)]
}

#==============================================================
# Loading circuit layout
#==============================================================

@st.cache_data
def load_track_map(track_name):
    import fastf1
    fastf1.Cache.enable_cache("cache")

    session = fastf1.get_session(2025, track_name, "R")
    session.load()

    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()

    # 🔥 FIX: use position data instead of telemetry
    pos = lap.get_pos_data()

    # Merge telemetry speed with position
    pos = pos.dropna(subset=['X', 'Y'])

    return pos["X"], pos["Y"]


# ==============================================================
# Load Calendar
# ==============================================================

@st.cache_data
def get_calendar():
    schedule = fastf1.get_event_schedule(2025)
    schedule = schedule.sort_values("RoundNumber")  
    return schedule["EventName"].tolist()


# ==============================================================
# LOAD PRECOMPUTED TRACK DATA
# ==============================================================

@st.cache_data
def load_track_data(track_name):

    path = f"tracks/{track_name}.pkl"

    if not os.path.exists(path):
        raise FileNotFoundError(f"{track_name} not precomputed")

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["df"], data["sc"], data["vsc"], data["deg"], data["laps"], data["track_params"]


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
    st.session_state.pit_lap = None


# ==============================================================
# UI SETUP
# ==============================================================

st.set_page_config(layout="wide")

st.title("🏎️ F1 Strategy Simulator")

calendar = get_calendar()

# Select track FIRST
track_name = st.sidebar.selectbox("Select Race", calendar)

# Load data for dropdown
df, _, _, _, _, _ = load_track_data(track_name)

# Driver dropdown WITH team
driver_options = [
    f"{row['Driver']} ({row['Team']})"
    for _, row in df.iterrows()
]

selected_driver_display = st.sidebar.selectbox("Select Driver", driver_options)

# Extract driver code
driver_choice = selected_driver_display.split(" ")[0]

# Tire selection
start_tire = st.sidebar.selectbox("Starting Tire", ["Soft", "Medium", "Hard"])

# Race speed (user-friendly)
sim_speed = st.sidebar.selectbox(
    "Race Speed",
    ["Slow", "Normal", "Fast", "Fast Forward"]
)

# Map to actual values
speed_map = {
    "Slow": (2, 50),
    "Normal": (5, 30),
    "Fast": (10, 20),
    "Fast Forward": (20, 10)
}

sim_speed, steps_per_lap = speed_map[sim_speed]

# Start button
start_button = st.sidebar.button("🚦 Start Race")


# ==============================================================
# START RACE
# ==============================================================

if start_button:

    # Load precomputed data
    df, sc, vsc, deg, laps, params = load_track_data(track_name)

    # Create environment
    env = F1Env(df, sc, vsc, deg)

    # ✅ SET TRACK LAPS (FIXED)
    env.total_laps = laps
    env.track_overtake_factor = params["overtake"]
    env.drs_zones = DRS_ZONES.get(track_name, [])

    state = env.reset()

    # Select player (for now first driver)
    player_driver = driver_choice
    player_idx = list(env.drivers).index(player_driver)

    # Set starting tire
    env.compounds[player_idx] = start_tire

    # Save state
    st.session_state.env = env
    st.session_state.state = state
    st.session_state.player_idx = player_idx
    st.session_state.running = True


# ==============================================================
# MAIN DISPLAY
# ==============================================================

if st.session_state.running:

    env = st.session_state.env
    state = st.session_state.state
    player_idx = st.session_state.player_idx

    col_left, col_center, col_right = st.columns([1, 3, 1])

    # ---------------- Center circuit view ----------------
    with col_center:
        st.subheader("🗺️ Track View")
        x, y = load_track_map(track_name)

        fig = go.Figure()

        # Track line
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='white', width=3),
            name='Track'
        ))

        # ---------------- DRS ZONES ----------------
        zones = DRS_ZONES.get(track_name, [])

        for start, end in zones:
            start_idx = int(start * (len(x) - 1))
            end_idx = int(end * (len(x) - 1))

            fig.add_trace(go.Scatter(
                x=x[start_idx:end_idx],
                y=y[start_idx:end_idx],
                mode='lines',
                line=dict(color='lime', width=6),
                opacity=0.4,
                showlegend=False
            ))

        sorted_idx = np.argsort(state["gaps"])

        for pos, i in enumerate(sorted_idx):
            

            driver = env.drivers[i]
            is_player = (i == player_idx)

            size = 16 if is_player else 10
            border_color = "yellow" if is_player else "black"
            border_width = 3 if is_player else 1

            progress = state["lap_progress"][i]

            idx = min(int(progress * (len(x) - 1)), len(x) - 1)

            # Safe indexing
            x_val = x.iloc[idx] if hasattr(x, "iloc") else x[idx]
            y_val = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

            # TEAM + COLOR
            team = df.loc[df["Driver"] == driver, "Team"].values[0] if "Team" in df.columns else "Unknown"
            color = TEAM_COLORS.get(team, "white")

            # GAP TO AHEAD
            if pos > 0:
                ahead_idx = sorted_idx[pos - 1]
                gap_ahead = state["gaps"][i] - state["gaps"][ahead_idx]
            else:
                gap_ahead = 0.0

            # HOVER TEXT
            hover_text = (
                f"<b>{driver}</b><br>"
                f"Team: {team}<br>"
                f"Position: {pos+1}<br>"
                f"Tire: {state['compound'][i]}<br>"
                f"Age: {int(state['tire_age'][i])}<br>"
                f"Gap Ahead: {gap_ahead:.2f}s"
            )

            fig.add_trace(go.Scatter(
             x=[x_val],
                y=[y_val],
                mode='markers',
                marker=dict(
                    size=size, 
                    color=color,
                    line=dict(width=border_width, color=border_color)),
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False
            ))

        fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            showlegend=False,
            height=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Left (Leaderboard) ---------------
    with col_left:
        st.subheader("🏁 Leaderboard")

        sorted_idx = np.argsort(state["gaps"])
        leader_time = state["gaps"][sorted_idx[0]]

        for pos, idx in enumerate(sorted_idx):
            driver = env.drivers[idx]
            gap = state["gaps"][idx] - leader_time

            st.write(
                f"P{pos+1} {driver} | "
                f"{'Leader' if pos==0 else f'+{gap:.2f}s'}"
            )

    # ---------------- RIGHT ( Driver Info) ---------------
    with col_right:
        st.subheader("📊 Driver")

        st.metric("Lap", f"{state['lap']} / {env.total_laps}")
        st.metric("Position", state["position"][player_idx] + 1)

        st.write(f"Tire: {state['compound'][player_idx]}")
        st.write(f"Age: {int(state['tire_age'][player_idx])}")

        sorted_idx = np.argsort(state["gaps"])
        pos = np.where(sorted_idx == player_idx)[0][0]

        if pos > 0:
            ahead = sorted_idx[pos - 1]
            gap_ahead = state["gaps"][player_idx] - state["gaps"][ahead]
            st.write(f"⬆ Gap Ahead: {gap_ahead:.2f}s")

        if pos < len(sorted_idx) - 1:
            behind = sorted_idx[pos + 1]
            gap_behind = state["gaps"][behind] - state["gaps"][player_idx]
            st.write(f"⬇ Gap Behind: {gap_behind:.2f}s")

# ==============================================================
# PIT SYSTEM (REAL-TIME)
# ==============================================================

    st.subheader("🛠️ Pit Strategy")

    colA, colB = st.columns([1, 2])

    with colA:
        if st.button("PIT NOW"):
            st.session_state.pit_requested = True
            st.session_state.pit_lap = state["lap"]

    with colB:
        st.session_state.pit_type = st.selectbox(
            "Select Tire",
            ["Soft", "Medium", "Hard"]
        )


# ==============================================================
# DETERMINE PLAYER PIT ACTION
# ==============================================================

    player_action = 0

    if st.session_state.pit_requested:

        current_lap = state["lap"]

        progress = state["lap_progress"][player_idx]

        # PIT ENTRY ZONE (end of lap)
        if progress > 0.7:
        
            player_action = {"Soft": 1, "Medium": 2, "Hard": 3}[st.session_state.pit_type]
            st.success("🟢 Entering pit THIS lap")
            st.session_state.pit_requested = False

        else:
            st.warning("⚠️ Not at pit entry yet — will pit next lap")

# ==============================================================
# AI DRIVERS
# ==============================================================

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


# ==============================================================
# STEP SIMULATION
# ==============================================================

    # STEP SIMULATION (ONE STEP PER RERUN)
    new_state, reward, done, info = env.step(actions)

    st.session_state.state = new_state


# ==============================================================
# ALERTS
# ==============================================================

    compound = new_state["compound"][player_idx]
    age = new_state["tire_age"][player_idx]

    if compound == "Soft":
        if age > 12:
            st.warning("🔴 Soft tire dropping off — PIT soon")

    elif compound == "Medium":
        if age > 22:
            st.warning("🟠 Medium tire degrading — consider pit")

    elif compound == "Hard":
        if age > 32:
            st.warning("🟡 Hard tire aging — strategy window")

    if new_state["safety"] == "SC":
        st.error("🚨 SAFETY CAR DEPLOYED — PIT WINDOW OPEN")

    elif new_state["safety"] == "VSC":
        st.warning("⚠️ VIRTUAL SAFETY CAR — PIT ADVANTAGE")

    if new_state["safety"] in ["SC", "VSC"]:
        st.info("💡 Strategy Tip: Consider pitting under Safety Car")

    # OVERTAKE ALERTS
    if "overtakes" in info:
        for attacker, defender in info["overtakes"]:
            st.success(f"⚡ {attacker} overtakes {defender}!")

# ==============================================================
# END RACE
# ==============================================================

    if done:
        st.success("🏁 Race Finished!")
        
        order = np.argsort(new_state["gaps"])

        results = pd.DataFrame({
            "Position": range(1, len(order) + 1),
            "Driver": env.drivers[order]
        })

        st.dataframe(results)

        st.session_state.running = False


# ==============================================================
# AUTO REFRESH (LIVE SIM)
# ==============================================================

    # control how long ONE LAP takes
    lap_time_seconds = {
        2: 10,   # Slow
        5: 7,    # Normal
        10: 5,   # Fast
        20: 2    # Fast Forward
    }

    sleep_time = lap_time_seconds.get(sim_speed, 5)

    time.sleep(sleep_time)
    st.rerun()