# Reinforcement Learning-based Multi-Agent Path Planning in POGEMA (rl-mapp-pogema)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![SKRL](https://img.shields.io/badge/SKRL-1.4.1%2B-green.svg)](https://github.com/Toni-SM/skrl)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository is designed for **mobile multi-robot navigation** using **Reinforcement Learning (RL)**. It supports both **single-agent** and **multi-agent** training in the [**POGEMA**](https://github.com/Cognitive-AI-Systems/pogema) environment, a grid-based world focused on decentralized pathfinding with partial observability.

We provide implementations of:
- **IPPO** (Independent Proximal Policy Optimization)
- **MAPPO** (Multi-Agent Proximal Policy Optimization)

These algorithms are implemented using the [**SKRL** framework](https://github.com/Toni-SM/skrl), enabling flexible experimentation with state-of-the-art RL approaches. 

## 🧭 Project Objective

This project implements reinforcement learning approaches for **decentralized multi-agent navigation**, where agents learn to reach their respective goals in a shared grid world. The focus is on:
- 🤝 Coordination  
- 🚫 Collision avoidance  
- 🧠 Efficient pathfinding under **partial observability**


