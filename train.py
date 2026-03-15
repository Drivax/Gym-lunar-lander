import gymnasium as gym
import numpy as np
import torch
import random
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from envs.wrappers import NoisyObservations, ActionDelayAware


ENV_ID = "LunarLander-v3"
SEED = 42
N_ENVS = 8

LOG_DIR = Path("logs")
MODEL_DIR = Path("models")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def make_env(rank, seed, noise_std, wind_power, turbulence_power, max_delay):
    def _init():
        env = gym.make(
            ENV_ID,
            enable_wind=wind_power > 0,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )
        env.reset(seed=seed + rank)

        env = ActionDelayAware(env, max_delay=max_delay)

        if noise_std > 0:
            env = NoisyObservations(env, noise_std=noise_std)

        return Monitor(env)
    return _init


def make_vec_env(n_envs, seed, noise_std, wind_power, turbulence_power, max_delay, vecnorm_path=None):

    env = SubprocVecEnv([
        make_env(i, seed, noise_std, wind_power, turbulence_power, max_delay)
        for i in range(n_envs)
    ])

    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=100.0,
        )

    return env


def create_model(env):

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        learning_rate = 7e-5,
        gae_lambda = 0.88,
        clip_range = 0.28,
        ent_coef = 0.003,
        vf_coef = 0.45,
        target_kl = 0.025,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.Tanh,
        ),
        tensorboard_log=str(LOG_DIR),
        seed=SEED,
        verbose=1,
    )

    return model


def train_phase(name, total_timesteps, noise_std, wind_power, turbulence_power, max_delay, model=None, vecnorm_path=None):

    print(f"\n===== {name} | {total_timesteps:,} steps =====")

    train_env = make_vec_env(N_ENVS, SEED, noise_std, wind_power, turbulence_power, max_delay, vecnorm_path)

    eval_env = make_vec_env(1, SEED + 1000, noise_std, wind_power, turbulence_power, max_delay, vecnorm_path)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / f"best_{name}"),
        log_path=str(LOG_DIR / name),
        eval_freq=max(25000 // N_ENVS, 1),
        deterministic=True,
        n_eval_episodes=20,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // N_ENVS, 1),
        save_path=str(MODEL_DIR / f"ckpt_{name}"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    if model is None:
        model = create_model(train_env)
    else:
        model.set_env(train_env)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    vecnorm_file = MODEL_DIR / f"vecnorm_{name}.pkl"
    train_env.save(vecnorm_file)

    train_env.close()
    eval_env.close()

    return model, vecnorm_file


if __name__ == "__main__":

    set_global_seed(SEED)

    configure(str(LOG_DIR), ["stdout", "tensorboard"])

    model, vec1 = train_phase("phase1", 5_000_000, 0, 0, 0, 0)

    model, vec2 = train_phase("phase2", 3_500_000, 0, 0, 0, 0, model, vec1)

    model, vec3 = train_phase("phase3", 2_500_000, 0, 2, 0, 0, model, vec2)

    model.save(MODEL_DIR / "ppo_lunar_lander_final")

    print("\nTraining finished")
