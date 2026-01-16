# ==================================================
# PPO LUNAR LANDER ROBUSTE – SCRIPT FINAL CORRIGÉ
# ==================================================

import gymnasium as gym
import numpy as np
from collections import deque
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from envs.wrappers import NoisyObservations, ActionDelayAware

# ==================================================
# 1. ENV FACTORY
# ==================================================

def make_env(
    n_envs,
    seed,
    noise_std,
    wind_power,
    turbulence_power,
    max_delay
):
    def _init():
        env = gym.make(
            "LunarLander-v3",
            render_mode=None,
            enable_wind=wind_power > 0,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )

        env = ActionDelayAware(env, max_delay=max_delay)

        if noise_std > 0:
            env = NoisyObservations(env, noise_std=noise_std)

        env = Monitor(env)
        return env

    vec_env = make_vec_env(
        _init,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=seed
    )

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,   # IMPORTANT pour LunarLander
        clip_obs=10.0,
        gamma=0.99
    )
    return vec_env


# ==================================================
# 3. LEARNING RATE SCHEDULE
# ==================================================

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


# ==================================================
# 4. TRAINING PHASE
# ==================================================

def train_phase(
    phase_name,
    total_timesteps,
    env_params,
    model=None
):
    print(f"\n===== PHASE {phase_name} =====")

    env = make_env(**env_params)

    if model is None:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=linear_schedule(2.5e-4),
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.92,
            clip_range=0.15,
            ent_coef=0.005,     # PAS de schedule ici
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256]
                ),
                activation_fn=torch.nn.Tanh,
                ortho_init=True,
            ),
            verbose=1,
            seed=42,
        )
    else:
        model.set_env(env)

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=10
    )

    return model, env


# ==================================================
# 5. MAIN (WINDOWS SAFE)
# ==================================================

if __name__ == "__main__":

    logger = configure("logs/ppo_lander_robust_final/", ["tensorboard"])

    N_ENVS = 8

    # -------------------------
    # PHASE 1 – PILOTAGE PROPRE
    # -------------------------
    model, env = train_phase(
        "1 – Sans perturbations",
        total_timesteps=1_500_000,
        env_params=dict(
            n_envs=N_ENVS,
            seed=42,
            noise_std=0.0,
            wind_power=0.0,
            turbulence_power=0.0,
            max_delay=0
        )
    )
    env.save("vecnorm_phase1.pkl")

    # -------------------------
    # PHASE 2 – PERTURBATIONS LÉGÈRES
    # -------------------------
    model, env = train_phase(
        "2 – Vent + bruit léger",
        total_timesteps=1_500_000,
        env_params=dict(
            n_envs=N_ENVS,
            seed=43,
            noise_std=0.04,
            wind_power=8.0,
            turbulence_power=0.8,
            max_delay=1
        ),
        model=model
    )
    env.save("vecnorm_phase2.pkl")

    # -------------------------
    # PHASE 3 – SETUP FINAL
    # -------------------------
    model, env = train_phase(
        "3 – Setup robuste final",
        total_timesteps=2_000_000,
        env_params=dict(
            n_envs=N_ENVS,
            seed=44,
            noise_std=0.08,
            wind_power=18.0,
            turbulence_power=1.8,
            max_delay=3
        ),
        model=model
    )
    env.save("vecnorm_phase3.pkl")

    model.save("ppo_lunar_lander_robust_final")

    print("\nENTRAÎNEMENT TERMINÉ")
    print("Modèle : ppo_lunar_lander_robust_final.zip")
