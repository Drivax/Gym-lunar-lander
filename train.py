import gymnasium as gym
import argparse
import torch
import numpy as np
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # backend hors écran
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

import seaborn as sns
import pandas as pd
from scipy.stats import entropy
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from scipy.stats import entropy



# =========================
# CONFIG
# =========================

ENV_ID = "LunarLander-v3"
SEED = 42
N_ENVS = 8

LOG_DIR = Path("logs")
MODEL_DIR = Path("models")

LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

GIF_DIR = Path("gifs")
GIF_DIR.mkdir(exist_ok=True)
# =========================
# SEED
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# ENV
# =========================

def make_env(rank, wind_power=0):

    def _init():

        env = gym.make(
            ENV_ID,
            enable_wind=wind_power > 0,
            wind_power=wind_power,
            turbulence_power=0.5 if wind_power > 0 else 0,
        )

        env.reset(seed=SEED + rank)

        return Monitor(env)

    return _init


def make_vec_env(n_envs, wind_power=0, vecnorm_path=None):

    env = SubprocVecEnv([
        make_env(i, wind_power) for i in range(n_envs)
    ])

    if vecnorm_path is None:

        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

    else:

        env = VecNormalize.load(vecnorm_path, env)
        env.training = True

    return env


# =========================
# MODEL
# =========================

def create_model(env):

    return PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,

        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        target_kl=0.03,

        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.Tanh,
        ),

        tensorboard_log=str(LOG_DIR),
        seed=SEED,
        verbose=1,
    )


# =========================
# PHASE TRAIN
# =========================

def train_baseline():

    print("\n===== PHASE 1 : BASELINE =====")

    env = make_vec_env(N_ENVS)

    eval_env = make_vec_env(1)
    eval_env.training = False
    eval_env.norm_reward = False

    model = create_model(env)

    callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_baseline",
        log_path="logs/baseline",
        eval_freq=25000 // N_ENVS,
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(
        total_timesteps=3_000_000,
        callback=callback,
        progress_bar=True,
    )

    env.save("models/vecnorm_baseline.pkl")

    model.save("models/ppo_baseline")

    env.close()
    eval_env.close()


# =========================
# PHASE FINE TUNING
# =========================

def fine_tune():

    print("\n===== PHASE 2 : FINE TUNE =====")

    env = make_vec_env(
        N_ENVS,
        wind_power=2,
        vecnorm_path="models/vecnorm_baseline.pkl",
    )

    eval_env = make_vec_env(
        1,
        wind_power=2,
        vecnorm_path="models/vecnorm_baseline.pkl",
    )

    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(
        "models/best_baseline/best_model.zip",
        env=env,
    )

    # PARAMS POLICY SHARPENING
    model.lr_schedule = get_schedule_fn(1e-4)
    model.gae_lambda = 0.88
    model.clip_range = get_schedule_fn(0.28)
    model.ent_coef = 0.003
    model.vf_coef = 0.45
    model.target_kl = 0.025

    callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_finetune",
        log_path="logs/finetune",
        eval_freq=25000 // N_ENVS,
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=callback,
        progress_bar=True,
    )

    model.save("models/ppo_final")

    env.close()
    eval_env.close()

def evaluate_model(
    model_path,
    vecnorm_path,
    wind_power=0,
    n_episodes=100,
    save_dir="advanced_eval",
):

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    env = make_vec_env(
        1,
        wind_power=wind_power,
        vecnorm_path=vecnorm_path,
    )

    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    all_positions = []
    all_velocities = []
    rewards = []
    lengths = []
    entropies = []
    kl_vals = []
    landing_types = []

    for ep in range(n_episodes):

        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        prev_dist = None

        while not done:

            action, _ = model.predict(obs, deterministic=False)

            obs_tensor = torch.as_tensor(obs).to(model.device)
            dist = model.policy.get_distribution(obs_tensor)

            ent = dist.distribution.entropy().mean().item()
            entropies.append(ent)

            if prev_dist is not None:
                kl = torch.distributions.kl_divergence(
                    prev_dist.distribution,
                    dist.distribution
                ).mean().item()
                kl_vals.append(kl)

            prev_dist = dist

            obs, reward, done_vec, info = env.step(action)

            if done_vec[0]:
                terminal_state = info[0].get("terminal_observation", obs[0])
                done = True
            else:
                done = False

            state_vec = obs[0]
            all_positions.append((state_vec[0], state_vec[1]))
            all_velocities.append((state_vec[2], state_vec[3]))

            ep_reward += reward[0]
            steps += 1

        terminal_state = info[0].get("terminal_observation", obs[0])
        landing_type = true_success_metric(
            terminal_state,
            True,
            ep_reward
        )
        landing_types.append(landing_type)

        rewards.append(ep_reward)
        lengths.append(steps)

    env.close()

    rewards = np.array(rewards)
    pos = np.array(all_positions)
    vel = np.array(all_velocities)

    # ========================
    # LANDING DISTRIBUTION
    # ========================

    from collections import Counter
    landing_counts = Counter(landing_types)

    plt.figure(figsize=(8,5))
    plt.bar(landing_counts.keys(), landing_counts.values())
    plt.title("Landing Quality Distribution")
    plt.xticks(rotation=30)
    plt.savefig(save_dir / "landing_quality.png")
    plt.close()

    # ========================
    # OTHER METRICS
    # ========================

    entropy_mean = float(np.mean(entropies))
    entropy_std = float(np.std(entropies))
    kl_mean = float(np.mean(kl_vals)) if len(kl_vals) > 0 else 0.0
    instability = float(np.std(rewards))

    mid = len(rewards)//2
    forgetting_score = float(rewards[:mid].mean() - rewards[mid:].mean())

    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "kl_mean": kl_mean,
        "instability": instability,
        "catastrophic_forgetting": forgetting_score,
        "landing_distribution": dict(landing_counts),
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n===== ADVANCED EVAL =====")
    for k, v in metrics.items():
        print(k, ":", v)

def landing_quality_distribution(
    model_path,
    vecnorm_path,
    wind_power=0,
    n_episodes=100,
):

    env = make_vec_env(
        1,
        wind_power=wind_power,
        vecnorm_path=vecnorm_path,
    )

    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    landing_results = []

    for ep in range(n_episodes):

        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = env.step(action)
            done = done_vec[0]
            ep_reward += reward[0]

        final_state = obs[0]

        landing_type = true_success_metric(
            final_state,
            done,
            ep_reward
        )

        landing_results.append(landing_type)

    env.close()

    # ======================
    # Distribution
    # ======================

    from collections import Counter
    counts = Counter(landing_results)

    print("\n===== LANDING QUALITY =====")
    for k, v in counts.items():
        print(k, ":", v, f"({v/n_episodes*100:.1f}%)")

    # ======================
    # Plot
    # ======================

    plt.figure(figsize=(8,5))
    plt.bar(counts.keys(), counts.values())
    plt.title("Landing Quality Distribution")
    plt.xticks(rotation=30)
    plt.show()

    return counts

def policy_diagnostic(
    model_path,
    vecnorm_path,
    wind_power=0,
    n_episodes=100,
    save_dir="diagnostic",
):

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    env = make_vec_env(
        1,
        wind_power=wind_power,
        vecnorm_path=vecnorm_path,
    )

    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    rewards = []
    action_history = []
    control_energy = []
    trajectory_lengths = []

    success_count = 0
    failure_count = 0
    neutral_count = 0

    for ep in range(n_episodes):

        obs = env.reset()
        done = False

        ep_reward = 0
        actions = []

        while not done:

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = env.step(action)
            done = done_vec[0]

            actions.append(action[0])
            ep_reward += reward[0]

        rewards.append(ep_reward)
        trajectory_lengths.append(len(actions))
        action_history.append(actions)

        actions = np.array(actions)
        control_energy.append(np.sum(actions**2))

        landing_type = true_success_metric(obs[0], done, ep_reward)

        if landing_type in ["PERFECT_LANDING", "GOOD_LANDING"]:
            success_count += 1
        elif landing_type == "CRASH":
            failure_count += 1
        else:
            neutral_count += 1


    env.close()

    rewards = np.array(rewards)
    control_energy = np.array(control_energy)

    # =========================
    # SUR-CONTROL
    # =========================

    energy_mean = control_energy.mean()
    surcontrol = energy_mean > 150

    # =========================
    # OSCILLATION FFT
    # =========================

    oscillation_scores = []

    for actions in action_history:
        a = np.array(actions)
        fft = np.abs(np.fft.rfft(a))
        high_freq = fft[int(len(fft)*0.5):].mean()
        oscillation_scores.append(high_freq)

    oscillation_index = np.mean(oscillation_scores)
    oscillates = oscillation_index > 0.5

    # =========================
    # LUCK vs OPTIMAL
    # =========================

    reward_std = rewards.std()
    optimal = (rewards.mean() > 250) and (reward_std < 80)

    # =========================
    # GENERALIZATION WIND
    # =========================

    wind_test_rewards = []

    for w in [0, 2, 4, 6]:

        test_env = make_vec_env(
            1,
            wind_power=w,
            vecnorm_path=vecnorm_path,
        )

        test_env.training = False
        test_env.norm_reward = False

        test_model = PPO.load(model_path, env=test_env)

        obs = test_env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = test_model.predict(obs, deterministic=True)
            obs, reward, done_vec, _ = test_env.step(action)
            done = done_vec[0]
            ep_reward += reward[0]

        wind_test_rewards.append(ep_reward)
        test_env.close()

    wind_generalization = np.std(wind_test_rewards) < 120

    # =========================
    # OVERFITTING
    # =========================

    overfit = rewards.mean() - np.mean(wind_test_rewards) > 80

    # =========================
    # BANG-BANG CONTROL
    # =========================

    bang_ratios = []

    for actions in action_history:
        actions = np.array(actions)
        bang = np.mean(np.abs(actions) > 0.8)
        bang_ratios.append(bang)

    bangbang = np.mean(bang_ratios) > 0.6

    # =========================
    # SAVE RESULTS
    # =========================

    diagnostics = {
        "episodes": n_episodes,
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "neutral_count": int(neutral_count),
        "success_rate": float(success_count / n_episodes),

        "surcontrol": bool(surcontrol),
        "oscillates": bool(oscillates),
        "optimal_policy": bool(optimal),
        "generalizes_wind": bool(wind_generalization),
        "overfit": bool(overfit),
        "bangbang_control": bool(bangbang),

        "mean_reward": float(rewards.mean()),
        "reward_std": float(reward_std),
        "control_energy": float(energy_mean),
        "oscillation_index": float(oscillation_index),
        "wind_rewards": [float(x) for x in wind_test_rewards],
    }

    with open(save_dir / "diagnostic.json", "w") as f:
        json.dump(diagnostics, f, indent=4)

    print("\n===== POLICY DIAGNOSTIC =====")
    for k, v in diagnostics.items():
        print(k, ":", v)
def true_success_metric(state, done, reward):

    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    angular_vel = state[5]
    left_contact = state[6]
    right_contact = state[7]

    dist_pad = np.sqrt(x**2 + y**2)

    # ======================
    # CRASH
    # ======================

    if done and reward < -50:
        return "CRASH"

    # ======================
    # PERFECT LANDING
    # ======================

    if (
        abs(vx) < 0.05
        and abs(vy) < 0.1
        and abs(angle) < 0.05
        and abs(angular_vel) < 0.1
        and dist_pad < 0.1
        and left_contact
        and right_contact
    ):
        return "PERFECT_LANDING"

    # ======================
    # GOOD LANDING
    # ======================

    if (
        abs(vx) < 0.1
        and abs(vy) < 0.2
        and abs(angle) < 0.1
        and abs(angular_vel) < 0.2
        and dist_pad < 0.2
    ):
        return "GOOD_LANDING"

    # ======================
    # HARD LANDING
    # ======================

    if (
        abs(vx) < 0.3
        and abs(vy) < 0.5
        and dist_pad < 0.3
    ):
        return "HARD_LANDING"

    # ======================
    # UNSTABLE
    # ======================

    return "UNSTABLE"
import imageio
import imageio

def record_policy_gif(
    model_path,
    vecnorm_path,
    wind_power=0,
    filename="run.gif",
    deterministic=True,
    max_steps=1000,
):

    env = gym.make(
        ENV_ID,
        enable_wind=wind_power > 0,
        wind_power=wind_power,
        turbulence_power=0.5 if wind_power > 0 else 0,
        render_mode="rgb_array",
    )

    model = PPO.load(model_path)

    # load normalization stats
    vecnorm = VecNormalize.load(vecnorm_path, SubprocVecEnv([lambda: Monitor(gym.make(ENV_ID))]))
    vecnorm.training = False
    vecnorm.norm_reward = False

    obs, _ = env.reset()

    frames = []

    for _ in range(max_steps):

        frame = env.render()
        frames.append(frame)

        obs_norm = vecnorm.normalize_obs(obs)

        action, _ = model.predict(obs_norm, deterministic=deterministic)

        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    env.close()

    imageio.mimsave(GIF_DIR / filename, frames, fps=30)
def trajectory_overlay_gif(model_path, n_episodes=3, save_dir="gifs"):
    from pathlib import Path
    import gymnasium as gym
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio.v3 as iio
    from stable_baselines3 import PPO

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    model = PPO.load(model_path, env=None)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        frames = []
        positions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame = env.render()
            frames.append(frame)
            positions.append(obs[:2])

        positions = np.array(positions)
        gif_frames = []

        for i, frame in enumerate(frames):
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(frame)
            traj_x = positions[:i+1,0]
            traj_y = positions[:i+1,1]
            x_norm = (traj_x - traj_x.min()) / (traj_x.max() - traj_x.min() + 1e-6) * frame.shape[1]
            y_norm = frame.shape[0] - (traj_y - traj_y.min()) / (traj_y.max() - traj_y.min() + 1e-6) * frame.shape[0]
            ax.plot(x_norm, y_norm, color='red', linewidth=2)
            ax.axis("off")
            fig.canvas.draw()

            # Conversion ARGB -> RGB compatible toutes versions
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (fig.canvas.get_width_height()[::-1][0],
                         fig.canvas.get_width_height()[::-1][1], 4)
            # ARGB -> RGB
            img = buf[:,:,[1,2,3]]
            plt.close(fig)
            gif_frames.append(img)

        gif_path = save_dir / f"trajectory_overlay_ep{ep:03d}.gif"
        iio.imwrite(gif_path, gif_frames, fps=15)
        print("Saved GIF:", gif_path)

    env.close()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate PPO on LunarLander with phased settings."
    )
    parser.add_argument(
        "--phase",
        choices=["baseline", "finetune", "advanced-eval", "diagnostic", "overlay-gif"],
        default="baseline",
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--model-path",
        default="models/ppo_final.zip",
        help="Model path for eval/diagnostic/gif phases.",
    )
    parser.add_argument(
        "--vecnorm-path",
        default="models/vecnorm_baseline.pkl",
        help="VecNormalize stats path for eval/diagnostic phases.",
    )
    parser.add_argument(
        "--wind-power",
        type=float,
        default=0.0,
        help="Wind power for evaluation phases.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for eval/diagnostic/gif phases.",
    )

    args = parser.parse_args()

    set_seed(SEED)

    if args.phase == "baseline":
        train_baseline()
    elif args.phase == "finetune":
        fine_tune()
    elif args.phase == "advanced-eval":
        evaluate_model(
            model_path=args.model_path,
            vecnorm_path=args.vecnorm_path,
            wind_power=args.wind_power,
            n_episodes=args.episodes,
        )
    elif args.phase == "diagnostic":
        policy_diagnostic(
            model_path=args.model_path,
            vecnorm_path=args.vecnorm_path,
            wind_power=args.wind_power,
            n_episodes=args.episodes,
        )
    elif args.phase == "overlay-gif":
        trajectory_overlay_gif(
            model_path=args.model_path,
            n_episodes=args.episodes,
            save_dir="gifs",
        )