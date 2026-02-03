import gymnasium as gym
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

from envs.wrappers import NoisyObservations, ActionDelayAware, ActionDelayAwareEval


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


USE_PERTURBATIONS = True
N_EPISODES = 50
MODEL_PATH = "models/ppo_lunar_lander_robust_final"
SAVE_GIF_EVERY = 10
GIF_PREFIX = "landing_perturbed" if USE_PERTURBATIONS else "landing_vanilla"

GIF_FOLDER = Path("gifs_eval")
GIF_FOLDER.mkdir(exist_ok=True)

env = make_eval_env(use_perturbations=USE_PERTURBATIONS)
model = PPO.load(MODEL_PATH)

trajectories = []
rewards = []
episode_lengths = []
successes = []

for i_episode in range(N_EPISODES):
    obs, _ = env.reset(seed=123 + i_episode * 7)

    frames = []
    positions = []
    velocities = []
    angles = []
    episode_reward = 0.0
    steps = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        steps += 1
        frames.append(env.render())

        positions.append((obs[0], obs[1]))
        velocities.append((obs[2], obs[3]))
        angles.append((obs[4], obs[5]))

    rewards.append(episode_reward)
    episode_lengths.append(steps)
    successes.append(episode_reward > 200)

    trajectories.append({
        "positions": positions,
        "velocities": velocities,
        "angles": angles,
        "reward": episode_reward,
        "length": steps,
        "success": episode_reward > 200,
        "perfect": episode_reward > 250,
        "crash": terminated and episode_reward < -50
    })

    if i_episode % SAVE_GIF_EVERY == 0:
        filename = GIF_FOLDER / f"{GIF_PREFIX}_ep{i_episode:03d}_r{episode_reward:.0f}.gif"
        iio.imwrite(filename, frames, fps=30, loop=0)
        print(f"GIF sauvegardé : {filename}")

env.close()


# ────────────────────────────────────────────────
# STATISTIQUES
# ────────────────────────────────────────────────

df = pd.DataFrame(trajectories)

print("\n" + "═" * 70)
print(" ÉVALUATION – LunarLander-v3 ".center(70))
print("═" * 70)

print(f"Épisodes              : {N_EPISODES}")
print(f"Reward moyen          : {df['reward'].mean():.2f} ± {df['reward'].std():.2f}")
print(f"Reward médian         : {df['reward'].median():.2f}")
print(f"Reward min / max      : {df['reward'].min():.2f} / {df['reward'].max():.2f}")
print(f"Durée moyenne         : {df['length'].mean():.1f} ± {df['length'].std():.1f} steps")
print(f"Taux de succès (>200) : {df['success'].mean():.2%}  ({df['success'].sum()}/{N_EPISODES})")
print(f"Taux parfait (>250)   : {df['perfect'].mean():.2%}")
print(f"Taux de crash         : {df['crash'].mean():.2%}  ({df['crash'].sum()}/{N_EPISODES})")

print("\nSuccès seulement :")
print(f"  → Reward moyen      : {df[df['success']]['reward'].mean():.2f}")
print(f"  → Durée moyenne     : {df[df['success']]['length'].mean():.1f} steps")

print("\nCrashes seulement :")
print(f"  → Reward moyen      : {df[df['crash']]['reward'].mean():.2f}")


# ────────────────────────────────────────────────
# VISUALISATIONS
# ────────────────────────────────────────────────

sns.set_style("whitegrid")

# 1. Trajectoires 2D superposées
plt.figure(figsize=(10, 8))
for traj in trajectories:
    pos = np.array(traj["positions"])
    color = "forestgreen" if traj["success"] else "indianred"
    alpha = 0.7 if traj["success"] else 0.4
    plt.plot(pos[:, 0], pos[:, 1], color=color, alpha=alpha, lw=1.2)

plt.axhspan(0, 0.1, color="gray", alpha=0.25, label="Zone atterrissage")
plt.title("Trajectoires 2D superposées")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.xlim(-1.5, 1.5)
plt.ylim(0, 1.5)
plt.grid(True, alpha=0.3)
plt.legend(["Succès", "Échec"], loc="upper right")
plt.savefig("trajectoires_2d.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Heatmap de densité des positions
plt.figure(figsize=(10, 8))
all_pos = np.vstack([np.array(t["positions"]) for t in trajectories])
sns.kdeplot(x=all_pos[:, 0], y=all_pos[:, 1], cmap="magma", fill=True, levels=12)
plt.title("Heatmap de densité des positions")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.xlim(-1.5, 1.5)
plt.ylim(0, 1.5)
plt.savefig("heatmap_positions.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Distribution des rewards
plt.figure(figsize=(10, 6))
sns.histplot(df["reward"], kde=True, bins=30, color="steelblue")
plt.axvline(200, color="green", linestyle="--", label="Seuil succès (200)")
plt.axvline(250, color="limegreen", linestyle="--", label="Parfait (250)")
plt.title("Distribution des récompenses")
plt.xlabel("Reward")
plt.ylabel("Nombre d'épisodes")
plt.legend()
plt.savefig("reward_distribution.png", dpi=150)
plt.close()

# 4. Boxplot reward par catégorie
plt.figure(figsize=(8, 6))
sns.boxplot(x="success", y="reward", data=df, palette=["salmon", "lightgreen"])
plt.title("Récompenses : Succès vs Échecs")
plt.xticks([0, 1], ["Échec/Crash", "Succès"])
plt.ylabel("Reward")
plt.savefig("boxplot_rewards.png", dpi=150)
plt.close()

print("\nVisualisations générées :")
print("• trajectoires_2d.png")
print("• heatmap_positions.png")
print("• reward_distribution.png")
print("• boxplot_rewards.png")