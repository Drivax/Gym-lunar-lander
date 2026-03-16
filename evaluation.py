import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.wrappers import NoisyObservations, ActionDelayAwareEval


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "models/ppo_baseline"
VECNORM_PATH = "models/vecnorm_phase3.pkl"

USE_PERTURBATIONS = False
N_EPISODES = 50

GIF_FOLDER = Path("gifs_eval")
GIF_FOLDER.mkdir(exist_ok=True)

SEED = 123


# =========================================================
# ENV FACTORY (ordre identique à l'entraînement)
# =========================================================

def make_env(seed):

    def _init():

        env = gym.make(
            "LunarLander-v3",
            render_mode="rgb_array",
            enable_wind=USE_PERTURBATIONS,
            wind_power=18.0 if USE_PERTURBATIONS else 0.0,
            turbulence_power=1.8 if USE_PERTURBATIONS else 0.0,
        )

        env.reset(seed=seed)

        # IMPORTANT : même wrapper EXACT
        env = ActionDelayAware(env, max_delay=3)

        if USE_PERTURBATIONS:
            env = NoisyObservations(env, noise_std=0.08)

        return env

    return _init



# =========================================================
# LOAD ENV + NORMALIZATION
# =========================================================

env = DummyVecEnv([make_env(SEED)])

env = VecNormalize.load(
    "models/vecnorm_phase3_robust.pkl",
    env
)
env.training = False
env.norm_reward = False

model = PPO.load(MODEL_PATH, env=env)


# =========================================================
# DEBUG CHECKS
# =========================================================

obs = env.reset()

print("\nEnvironment check")
print("-------------------")
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)


# =========================================================
# ROLLOUT
# =========================================================

trajectories = []
rewards = []
lengths = []

for ep in range(N_EPISODES):

    obs = env.reset()

    done = False
    episode_reward = 0
    steps = 0

    positions = []
    velocities = []
    angles = []

    frames = []

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        episode_reward += reward[0]
        steps += 1

        frame = env.render()
        frames.append(frame)

        state = obs[0][:8]

        positions.append((state[0], state[1]))
        velocities.append((state[2], state[3]))
        angles.append((state[4], state[5]))

    rewards.append(episode_reward)
    lengths.append(steps)

    trajectories.append({
        "positions": positions,
        "velocities": velocities,
        "angles": angles,
        "reward": episode_reward,
        "length": steps,
        "success": episode_reward > 200,
        "perfect": episode_reward > 250,
        "crash": episode_reward < -50
    })

    if ep % 10 == 0:
        import imageio.v3 as iio
        gif_path = GIF_FOLDER / f"episode_{ep:03d}.gif"
        iio.imwrite(gif_path, frames, fps=30)
        print("GIF saved:", gif_path)


env.close()


# =========================================================
# STATISTICS
# =========================================================

df = pd.DataFrame(trajectories)

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

print(f"Episodes              : {N_EPISODES}")
print(f"Reward mean           : {df.reward.mean():.2f} ± {df.reward.std():.2f}")
print(f"Reward median         : {df.reward.median():.2f}")
print(f"Reward min / max      : {df.reward.min():.2f} / {df.reward.max():.2f}")
print(f"Episode length mean   : {df.length.mean():.1f}")
print(f"Success rate (>200)   : {(df.success.mean()*100):.2f}%")
print(f"Perfect landings      : {(df.perfect.mean()*100):.2f}%")
print(f"Crash rate            : {(df.crash.mean()*100):.2f}%")


# =========================================================
# VISUALIZATIONS
# =========================================================

sns.set_style("whitegrid")

# trajectory plot
plt.figure(figsize=(10,8))

for traj in trajectories:

    pos = np.array(traj["positions"])

    color = "green" if traj["success"] else "red"

    plt.plot(pos[:,0], pos[:,1], alpha=0.5, color=color)

plt.title("LunarLander trajectories")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-1.5,1.5)
plt.ylim(0,1.5)

plt.savefig("trajectories.png")
plt.close()


# reward distribution
plt.figure(figsize=(8,6))

sns.histplot(df.reward, bins=30, kde=True)

plt.axvline(200, linestyle="--")

plt.title("Reward distribution")

plt.savefig("reward_distribution.png")
plt.close()


# heatmap positions
plt.figure(figsize=(10,8))

all_pos = np.vstack([np.array(t["positions"]) for t in trajectories])

sns.kdeplot(x=all_pos[:,0], y=all_pos[:,1], fill=True)

plt.title("Position heatmap")

plt.savefig("heatmap_positions.png")
plt.close()


print("\nFigures generated:")
print(" - trajectories.png")
print(" - reward_distribution.png")
print(" - heatmap_positions.png")
