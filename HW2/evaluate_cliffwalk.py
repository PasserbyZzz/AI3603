# -*- coding:utf-8 -*-
"""
Evaluation and visualization utility for CliffWalking (Sarsa, Q-Learning, Dyna-Q).
Functions:
- Train each algorithm (no rendering) and persist results to pickles
- Plot comparison curves for rewards and epsilon
- Plot per-algorithm curves
- Visualize greedy final path and save per-algorithm path images

Notes:
- Uses the old Gym API (gym==0.21)
- Intentionally different structure/naming from any other visualization utility
"""
import os
import pickle
import random
from typing import Dict, Tuple

import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # non-interactive backend for file saving
import matplotlib.pyplot as plt
# Set a global plotting style (choose a widely available style)
from matplotlib.patches import Rectangle
import gym

from agent import SarsaAgent, QLearningAgent, Dyna_QAgent


# ------------------------------
# Configuration & small helpers
# ------------------------------

GRID_R, GRID_C = 4, 12
START_S, GOAL_S = 36, 47


def moving_average(x: np.ndarray, k: int = 20) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return x
    k = min(k, len(x))
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode="valid")


def persist_results(name: str, rewards: np.ndarray, eps: np.ndarray, agent) -> None:
    """Serialize training traces and a dense Q-matrix to avoid pickling
    issues with defaultdict(lambda: ...). The stored `Q_table` is a
    numpy array of shape (max_state+1, n_actions) suitable for external
    visualization utilities that index like Q[state, :].
    """
    # Build a dense Q matrix
    try:
        n_actions = len(agent.all_actions)
    except Exception:
        # Fallback: infer from one row
        any_row = next(iter(agent.Q.values()))
        n_actions = len(any_row)
    max_state = max(agent.Q.keys()) if len(agent.Q) > 0 else 0
    Q_mat = np.zeros((max_state + 1, n_actions), dtype=np.float32)
    for s, row in agent.Q.items():
        Q_mat[int(s), :len(row)] = np.asarray(row, dtype=np.float32)

    payload = {
        "episode_rewards": np.asarray(rewards, dtype=float),
        "epsilon_history": np.asarray(eps, dtype=float),
        "Q_table": Q_mat,
    }
    with open(f"data/{name}_data.pkl", "wb") as f:
        pickle.dump(payload, f)


def read_results(name: str):
    try:
        with open(f"data/{name}_data.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"[WARN] {name}_data.pkl not found")
        return None


def greedy_argmax(row: np.ndarray) -> int:
    m = np.max(row)
    idx = np.where(row == m)[0]
    return int(np.random.choice(idx))


# ------------------------------
# Training routines
# ------------------------------

def run_sarsa(env, episodes: int, action_space: np.ndarray):
    agent = SarsaAgent(action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995)
    rewards, eps = [], []
    for _ in range(episodes):
        s = env.reset()
        a = agent.choose_action(s)
        total = 0
        for _ in range(500):
            s2, r, done, _ = env.step(a)
            total += r
            if done:
                agent.learn(s, a, r, s2, a_next=None, done=True)
                break
            a2 = agent.choose_action(s2)
            agent.learn(s, a, r, s2, a_next=a2, done=False)
            s, a = s2, a2
        rewards.append(total)
        # Per-episode epsilon decay (instead of per-step)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        eps.append(agent.epsilon)
    return agent, np.array(rewards), np.array(eps)


def run_offpolicy(env, agent, episodes: int):
    rewards, eps = [], []
    for _ in range(episodes):
        s = env.reset()
        total = 0
        for _ in range(500):
            a = agent.choose_action(s)
            s2, r, done, _ = env.step(a)
            total += r
            agent.learn(s, a, r, s2, done=done)
            s = s2
            if done:
                break
        rewards.append(total)
        # Per-episode epsilon decay (instead of per-step)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        eps.append(agent.epsilon)
    return agent, np.array(rewards), np.array(eps)


# ------------------------------
# Plotting utilities
# ------------------------------

def compare_reward_curves(names=("sarsa", "qlearning", "dyna_q"), window: int = 20):
    color_map = {"sarsa": "tab:blue", "qlearning": "tab:red", "dyna_q": "tab:green"}
    label_map = {"sarsa": "Sarsa", "qlearning": "Q-Learning", "dyna_q": "Dyna-Q"}
    plt.figure(figsize=(12, 6))
    for n in names:
        data = read_results(n)
        if not data:
            continue
        r = np.asarray(data["episode_rewards"], dtype=float)
        plt.plot(r, color=color_map[n], alpha=0.25, linewidth=0.6)
        plt.plot(moving_average(r, window), color=color_map[n], label=label_map[n], linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/comparison_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()


def compare_eps_curves(names=("sarsa", "qlearning", "dyna_q")):
    color_map = {"sarsa": "tab:blue", "qlearning": "tab:red", "dyna_q": "tab:green"}
    label_map = {"sarsa": "Sarsa", "qlearning": "Q-Learning", "dyna_q": "Dyna-Q"}
    plt.figure(figsize=(12, 6))
    for n in names:
        data = read_results(n)
        if not data:
            continue
        e = np.asarray(data["epsilon_history"], dtype=float)
        plt.plot(e, color=color_map[n], label=label_map[n], linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon (ε)")
    plt.title("Epsilon Decay Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/comparison_epsilon.png", dpi=300, bbox_inches="tight")
    plt.close()


def single_algo_panels(name: str, window: int = 20):
    data = read_results(name)
    if not data:
        return
    label_map = {"sarsa": "Sarsa", "qlearning": "Q-Learning", "dyna_q": "Dyna-Q"}
    rewards = np.asarray(data["episode_rewards"], dtype=float)
    eps = np.asarray(data["epsilon_history"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # rewards
    axes[0].plot(rewards, color="gray", alpha=0.35, linewidth=0.7)
    axes[0].plot(moving_average(rewards, window), color="tab:blue", linewidth=2.0)
    axes[0].set_title(f"{label_map[name]} — Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, alpha=0.3)
    # epsilon
    axes[1].plot(eps, color="orange", linewidth=2.0)
    axes[1].set_title(f"{label_map[name]} — Epsilon")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Epsilon (ε)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{name}_training.png", dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------
# Path visualization
# ------------------------------

def rollout_greedy(env, agent, limit: int = 60):
    # greedy evaluation: use Q-table argmax directly
    s = env.reset()
    path = [s]
    for _ in range(limit):
        qrow = agent.Q[s]
        a = greedy_argmax(qrow)
        s, r, done, _ = env.step(a)
        path.append(s)
        if done:
            break
    return path


def draw_path(path: np.ndarray, filename: str, title: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    rows, cols = GRID_R, GRID_C
    # grid lines
    for r in range(rows + 1):
        ax.plot([0, cols], [r, r], "k-", linewidth=1)
    for c in range(cols + 1):
        ax.plot([c, c], [0, rows], "k-", linewidth=1)
    # cliff cells (bottom row, 1..10)
    for c in range(1, cols - 1):
        ax.add_patch(Rectangle((c, 0), 1, 1, facecolor="red", alpha=0.3, edgecolor="darkred", linewidth=2))
        ax.text(c + 0.5, 0.5, "X", ha="center", va="center", fontsize=13, color="darkred")
    # start/goal
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="green", alpha=0.3, edgecolor="darkgreen", linewidth=2))
    ax.text(0.5, 0.5, "S", ha="center", va="center", fontsize=15, color="darkgreen")
    ax.add_patch(Rectangle((cols - 1, 0), 1, 1, facecolor="gold", alpha=0.3, edgecolor="orange", linewidth=2))
    ax.text(cols - 0.5, 0.5, "G", ha="center", va="center", fontsize=15, color="orange")

    # draw path arrows
    def s2rc(s):
        return s // cols, s % cols
    pts = []
    for s in path:
        r, c = s2rc(s)
        x, y = c + 0.5, (rows - 1 - r) + 0.5
        pts.append((x, y))
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.2, head_length=0.15, fc="blue", ec="blue", linewidth=2, alpha=0.7)
    # points
    for i, (x, y) in enumerate(pts):
        if i == 0:
            ax.plot(x, y, "go", markersize=11, markeredgecolor="darkgreen", markeredgewidth=2)
        elif i == len(pts) - 1:
            ax.plot(x, y, "y*", markersize=14, markeredgecolor="orange", markeredgewidth=2)
        else:
            ax.plot(x, y, "bo", markersize=6, alpha=0.6)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.invert_yaxis()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------
# Orchestration
# ------------------------------

if __name__ == "__main__":
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Build environments with identical seeds
    env_sarsa = gym.make("CliffWalking-v0")
    env_q = gym.make("CliffWalking-v0")
    env_d = gym.make("CliffWalking-v0")
    env_eval_s = gym.make("CliffWalking-v0")
    env_eval_q = gym.make("CliffWalking-v0")
    env_eval_d = gym.make("CliffWalking-v0")
    for e in [env_sarsa, env_q, env_d, env_eval_s, env_eval_q, env_eval_d]:
        e.seed(RANDOM_SEED)

    nA = env_sarsa.action_space.n
    A = np.arange(nA)
    EPISODES = 1000

    # Train and persist
    s_agent, s_rewards, s_eps = run_sarsa(env_sarsa, EPISODES, A)
    persist_results("sarsa", s_rewards, s_eps, s_agent)

    q_agent = QLearningAgent(A, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995)
    q_agent, q_rewards, q_eps = run_offpolicy(env_q, q_agent, EPISODES)
    persist_results("qlearning", q_rewards, q_eps, q_agent)

    d_agent = Dyna_QAgent(A, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, n_planning=20)
    d_agent, d_rewards, d_eps = run_offpolicy(env_d, d_agent, EPISODES)
    persist_results("dyna_q", d_rewards, d_eps, d_agent)

    # Generate visuals
    for name in ("sarsa", "qlearning", "dyna_q"):
        single_algo_panels(name)
    compare_reward_curves()
    compare_eps_curves()

    # Path images
    s_path = rollout_greedy(env_eval_s, s_agent)
    q_path = rollout_greedy(env_eval_q, q_agent)
    d_path = rollout_greedy(env_eval_d, d_agent)
    draw_path(s_path, "sarsa_path.png", "Sarsa — Learned Path")
    draw_path(q_path, "qlearning_path.png", "Q-Learning — Learned Path")
    draw_path(d_path, "dyna_q_path.png", "Dyna-Q — Learned Path")

    print("Artifacts saved: comparison_rewards.png, comparison_epsilon.png, *_training.png, *_path.png and *_data.pkl")
