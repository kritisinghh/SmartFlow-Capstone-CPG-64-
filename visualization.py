import os
import sys
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import traci

# =======================================
# PATH SETUP
# =======================================
BASE = r"C:\Users\ragha\OneDrive\Desktop\capy new"
sys.path.append(BASE)

from env import TrafficEnv
from mappo import MAPPO_GNN
from utils import get_average_travel_time, get_average_CO2, get_average_fuel

MODEL_PATH  = rf"{BASE}\results\trained_model1000.th"
SUMO_CFG    = rf"{BASE}\amman.sumocfg"
RESULTS_DIR = rf"{BASE}\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

K = 5
SEED = 42

N_AGENTS, OBS_DIM, ACT_DIM = 2, 10, 2

# =======================================
# RUN RL
# =======================================
def run_rl_once(run_id):
    env = TrafficEnv(mode='binary')
    env.sumoCmd[2] = SUMO_CFG
    env.sumoCmd += ["--tripinfo-output", rf"{BASE}\amman.tripinfo.xml", "--seed", str(SEED)]

    agent = MAPPO_GNN(N_AGENTS, OBS_DIM, ACT_DIM)
    agent.load_model(MODEL_PATH)

    state, done = env.reset(), False
    while not done:
        actions = []
        for i in range(N_AGENTS):
            a, _, _ = agent.select_action(state[i])
            actions.append(a)
        state, _, done = env.step(actions)

    env.close()
    time.sleep(0.5)

    src = rf"{BASE}\amman.tripinfo.xml"
    dst = rf"{BASE}\amman_rl_{run_id}.tripinfo.xml"
    if os.path.exists(src):
        shutil.copy(src, dst)
    return dst

# =======================================
# RUN STATIC
# =======================================
def run_static_once(run_id):
    env = TrafficEnv(mode='binary')
    env.sumoCmd[2] = SUMO_CFG
    env.sumoCmd += ["--tripinfo-output", rf"{BASE}\amman.tripinfo.xml", "--seed", str(SEED)]

    env.reset()
    done = False
    while not done:
        traci.simulationStep()
        if traci.simulation.getMinExpectedNumber() <= 0:
            done = True

    env.close()
    time.sleep(0.5)

    src = rf"{BASE}\amman.tripinfo.xml"
    dst = rf"{BASE}\amman_static_{run_id}.tripinfo.xml"
    if os.path.exists(src):
        shutil.copy(src, dst)
    return dst

# =======================================
# MATH HELPERS
# =======================================
def read_metrics(file):
    tmp = rf"{BASE}\amman.tripinfo.xml"
    shutil.copy(file, tmp)
    return round(get_average_travel_time(),2), round(get_average_CO2(),2), round(get_average_fuel(),2)

def mean_std(v):
    arr = np.array(v)
    return arr.mean(), arr.std(ddof=1) if len(v) > 1 else 0.0

def percent_improve(rl, st):
    return round((st - rl) / st * 100, 2)

# =======================================
# PLOT MAIN BAR CHART
# =======================================
def plot_main_bars(rl_means, rl_stds, st_means, st_stds, improvements, metrics):
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(10,6))

    b1 = plt.bar(x-width/2, rl_means, width, yerr=rl_stds, capsize=6,
                 color='green', label="MAPPO-GNN (RL)", alpha=0.85)
    b2 = plt.bar(x+width/2, st_means, width, yerr=st_stds, capsize=6,
                 color='gray', label="Static Baseline", alpha=0.85)

    for i, bar in enumerate(b1):
        y = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, y+0.02*y,
                 f"+{improvements[i]}%", ha='center', fontsize=12,
                 fontweight='bold', color='green')

    plt.xticks(x, metrics, fontsize=12)
    plt.ylabel("Mean ± Std", fontsize=12)
    plt.title(f"Performance Comparison: RL vs Static ({K} runs)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =======================================
# SEPARATE BAR GRAPHS
# =======================================
def plot_separate(rl_means, st_means, metrics):
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(6,4))
        plt.bar(["RL", "Static"], [rl_means[i], st_means[i]],
                color=['green','gray'], alpha=0.85)
        plt.ylabel(metric)
        plt.title(f"RL vs Static: {metric}")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

# =======================================
# MAIN
# =======================================
if __name__ == "__main__":
    rl_tt=[]; rl_co2=[]; rl_fuel=[]
    st_tt=[]; st_co2=[]; st_fuel=[]

    for i in range(1, K+1):
        print(f"Run {i}/{K}")
        rl_file = run_rl_once(i)
        st_file = run_static_once(i)

        tt, co2, fuel = read_metrics(rl_file)
        rl_tt.append(tt); rl_co2.append(co2); rl_fuel.append(fuel)

        tt, co2, fuel = read_metrics(st_file)
        st_tt.append(tt); st_co2.append(co2); st_fuel.append(fuel)

    metrics = ["Travel Time (s)", "CO₂ (g/km)", "Fuel (L/100km)"]

    rl_means = [mean_std(rl_tt)[0], mean_std(rl_co2)[0], mean_std(rl_fuel)[0]]
    rl_stds  = [mean_std(rl_tt)[1], mean_std(rl_co2)[1], mean_std(rl_fuel)[1]]
    st_means = [mean_std(st_tt)[0], mean_std(st_co2)[0], mean_std(st_fuel)[0]]
    st_stds  = [mean_std(st_tt)[1], mean_std(st_co2)[1], mean_std(st_fuel)[1]]

    improvements = [
        percent_improve(rl_means[0], st_means[0]),
        percent_improve(rl_means[1], st_means[1]),
        percent_improve(rl_means[2], st_means[2])
    ]

    plot_main_bars(rl_means, rl_stds, st_means, st_stds, improvements, metrics)
    plot_separate(rl_means, st_means, metrics)
