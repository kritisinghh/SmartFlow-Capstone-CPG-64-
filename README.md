# SmartFlow-Capstone-CPG-64

# SmartFlow: Adaptive Traffic Signal Control Using MAPPO + GNN
SmartFlow is an AI-driven adaptive traffic signal control system designed to reduce congestion, fuel consumption, and CO₂ emissions in urban traffic networks.
The system uses Multi-Agent Proximal Policy Optimization (MAPPO) combined with Graph Neural Networks (GNNs) to enable intelligent, cooperative decision-making across multiple intersections simulated in SUMO.

# Project Overview
Conventional traffic signals follow fixed-time cycles and cannot adapt to real-time traffic conditions.
SmartFlow solves this by:
Treating every intersection as an intelligent RL agent
Allowing agents to communicate through a graph structure (road network)
Learning optimal signal timings that minimize vehicle waiting time, fuel usage, and CO₂ emissions
Running entirely on simulation, so no physical sensors are required
This results in smoother traffic flow, reduced emissions, and improved network efficiency.

# Key Features
MAPPO-based Multi-Agent RL: Centralized training with decentralized execution
GNN Integration: Each agent leverages neighborhood information for cooperative learning
SUMO Simulation: Realistic traffic environment with custom network files
Performance Metrics: Travel time, vehicle delay, fuel consumption, CO₂ emissions
Visualization Tools: Graphs showing comparisons with static/fixed-time control
Extensible Codebase: Easy to modify for decentralized variants or sensor-based deployment
