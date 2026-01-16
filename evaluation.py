import gymnasium as gym
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from collections import deque
from envs.wrappers import NoisyObservations, ActionDelayAware,ActionDelayAwareEval


# ────────────────────────────────────────────────
# ENV FACTORY
# ────────────────────────────────────────────────

def make_eval_env(use_perturbations=True, seed=123):
    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array",
        enable_wind=use_perturbations,
        wind_power=18.0 if use_perturbations else 0.0,
        turbulence_power=1.8 if use_perturbations else 0.0,
    )

    if use_perturbations:
        env = NoisyObservations(env, noise_std=0.08)
        env = ActionDelayAwareEval(env, max_delay=3)

    return env


# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

USE_PERTURBATIONS = True
N_EPISODES = 20
MODEL_PATH = "models/ppo_lunar_lander_robust_final"
SAVE_GIF_EVERY = 10
GIF_PREFIX = "landing_perturbed" if USE_PERTURBATIONS else "landing_vanilla"

env = make_eval_env(USE_PERTURBATIONS)
model = PPO.load(MODEL_PATH)

trajectories = []
rewards = []

# ────────────────────────────────────────────────
# ROLLOUTS
# ────────────────────────────────────────────────

for i_episode in range(N_EPISODES):
    obs, _ = env.reset(seed=123 + i_episode * 7)

    frames = []
    positions, velocities, angles = [], [], []
    episode_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        frames.append(env.render())

        # OBS = [0..7] env + [8] last_action
        positions.append((obs[0], obs[1]))
        velocities.append((obs[2], obs[3]))
        angles.append((obs[4], obs[5]))

    rewards.append(episode_reward)
    success = episode_reward > 200

    trajectories.append({
        "positions": positions,
        "velocities": velocities,
        "angles": angles,
        "reward": episode_reward,
        "success": success,
    })

    if i_episode % SAVE_GIF_EVERY == 0:
        filename = f"{GIF_PREFIX}_ep{i_episode:03d}_reward{episode_reward:.0f}.gif"
        iio.imwrite(filename, frames, fps=30, loop=0)
        print(f"→ GIF sauvegardé : {filename}")

env.close()

# ────────────────────────────────────────────────
# STATS
# ────────────────────────────────────────────────

rewards_array = np.array(rewards)
print(f"Mean reward: {rewards_array.mean():.2f} ± {rewards_array.std():.2f}")
print(f"Success rate: {(rewards_array > 200).mean():.2%}")

# ────────────────────────────────────────────────
# VISUALISATIONS
# ────────────────────────────────────────────────

# Trajectoires 2D
plt.figure(figsize=(10, 8))
for traj in trajectories:
    pos = np.array(traj["positions"])
    color = "green" if traj["success"] else "red"
    plt.plot(pos[:, 0], pos[:, 1], color=color, alpha=0.4)

plt.axhspan(0, 0.1, color="gray", alpha=0.3)
plt.title("Trajectoires 2D superposées")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-1.5, 1.5)
plt.ylim(0, 1.5)
plt.grid(True)
plt.savefig("trajectoires_2d.png")
plt.show()

# Heatmap
all_pos = np.vstack([np.array(t["positions"]) for t in trajectories])
sns.kdeplot(x=all_pos[:, 0], y=all_pos[:, 1], cmap="viridis", fill=True)
plt.title("Heatmap densité positions")
plt.xlim(-1.5, 1.5)
plt.ylim(0, 1.5)
plt.savefig("heatmap_positions.png")
plt.show()
