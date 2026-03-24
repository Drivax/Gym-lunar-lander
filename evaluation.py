import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.wrappers import ActionDelayAware, NoisyObservations


def build_env(seed: int, use_perturbations: bool, wind_power: float, max_delay: int, noise_std: float):
    """Create the evaluation environment with wrappers in the same order as training."""

    def _init():
        env = gym.make(
            "LunarLander-v3",
            render_mode="rgb_array",
            enable_wind=use_perturbations,
            wind_power=wind_power if use_perturbations else 0.0,
            turbulence_power=0.5 * wind_power if use_perturbations else 0.0,
        )
        env.reset(seed=seed)

        if use_perturbations:
            env = ActionDelayAware(env, max_delay=max_delay)
            env = NoisyObservations(env, noise_std=noise_std)

        return env

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO LunarLander policy.")
    parser.add_argument("--model-path", default="models/ppo_final.zip", help="Path to model zip file.")
    parser.add_argument(
        "--vecnorm-path",
        default="models/vecnorm_baseline.pkl",
        help="Path to VecNormalize stats file.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument(
        "--perturbations",
        action="store_true",
        help="Enable wind, delay, and noisy observations during evaluation.",
    )
    parser.add_argument("--wind-power", type=float, default=18.0, help="Wind power when perturbations are enabled.")
    parser.add_argument("--max-delay", type=int, default=3, help="Max action delay when perturbations are enabled.")
    parser.add_argument("--noise-std", type=float, default=0.08, help="Observation noise std when perturbations are enabled.")
    parser.add_argument("--gif-every", type=int, default=10, help="Save a GIF every N episodes.")
    parser.add_argument("--output-dir", default="gifs_eval", help="Directory for GIFs and plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    vec_env = DummyVecEnv([
        build_env(
            seed=args.seed,
            use_perturbations=args.perturbations,
            wind_power=args.wind_power,
            max_delay=args.max_delay,
            noise_std=args.noise_std,
        )
    ])

    env = VecNormalize.load(args.vecnorm_path, vec_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.model_path, env=env)

    trajectories = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        positions = []
        velocities = []
        angles = []
        frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, _ = env.step(action)
            done = bool(done_vec[0])

            episode_reward += float(reward[0])
            steps += 1

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            state = obs[0][:8]
            positions.append((float(state[0]), float(state[1])))
            velocities.append((float(state[2]), float(state[3])))
            angles.append((float(state[4]), float(state[5])))

        trajectories.append(
            {
                "positions": positions,
                "velocities": velocities,
                "angles": angles,
                "reward": episode_reward,
                "length": steps,
                "success": episode_reward > 200,
                "perfect": episode_reward > 250,
                "crash": episode_reward < -50,
            }
        )

        if args.gif_every > 0 and ep % args.gif_every == 0 and frames:
            gif_path = output_dir / f"episode_{ep:03d}.gif"
            iio.imwrite(gif_path, frames, fps=30)
            print(f"Saved GIF: {gif_path}")

    env.close()

    df = pd.DataFrame(trajectories)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes              : {args.episodes}")
    print(f"Reward mean           : {df.reward.mean():.2f} ± {df.reward.std():.2f}")
    print(f"Reward median         : {df.reward.median():.2f}")
    print(f"Reward min / max      : {df.reward.min():.2f} / {df.reward.max():.2f}")
    print(f"Episode length mean   : {df.length.mean():.1f}")
    print(f"Success rate (>200)   : {(df.success.mean() * 100):.2f}%")
    print(f"Perfect landings      : {(df.perfect.mean() * 100):.2f}%")
    print(f"Crash rate            : {(df.crash.mean() * 100):.2f}%")

    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 8))
    for traj in trajectories:
        pos = np.array(traj["positions"])
        if len(pos) == 0:
            continue
        color = "green" if traj["success"] else "red"
        plt.plot(pos[:, 0], pos[:, 1], alpha=0.5, color=color)

    plt.title("LunarLander trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 1.5)
    plt.savefig(output_dir / "trajectories.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(df.reward, bins=30, kde=True)
    plt.axvline(200, linestyle="--")
    plt.title("Reward distribution")
    plt.savefig(output_dir / "reward_distribution.png")
    plt.close()

    all_pos = [np.array(t["positions"]) for t in trajectories if len(t["positions"]) > 0]
    if all_pos:
        stacked_pos = np.vstack(all_pos)
        plt.figure(figsize=(10, 8))
        sns.kdeplot(x=stacked_pos[:, 0], y=stacked_pos[:, 1], fill=True)
        plt.title("Position heatmap")
        plt.savefig(output_dir / "heatmap_positions.png")
        plt.close()

    summary_path = output_dir / "evaluation_summary.csv"
    df.drop(columns=["positions", "velocities", "angles"]).to_csv(summary_path, index=False)

    print("\nFigures generated:")
    print(f" - {output_dir / 'trajectories.png'}")
    print(f" - {output_dir / 'reward_distribution.png'}")
    print(f" - {output_dir / 'heatmap_positions.png'}")
    print(f"Saved tabular summary: {summary_path}")


if __name__ == "__main__":
    main()
