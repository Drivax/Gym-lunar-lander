# ==================================================
# PPO LUNAR LANDER ROBUSTE – VERSION AMÉLIORÉE
# ==================================================

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from envs.wrappers import NoisyObservations, ActionDelayAware

# ==================================================
# ENV FACTORY
# ==================================================

def make_single_env(
    seed: int,
    noise_std: float,
    wind_power: float,
    turbulence_power: float,
    max_delay: int,
):
    def _init():
        env = gym.make(
            "LunarLander-v3",
            render_mode=None,
            enable_wind=wind_power > 0.0,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )

        env.reset(seed=seed)


        env = ActionDelayAware(env, max_delay=max_delay)

        if noise_std > 0:
            env = NoisyObservations(env, noise_std=noise_std)

        return Monitor(env)

    return _init


def make_vec_env_robust(
    n_envs: int,
    seed: int,
    noise_std: float,
    wind_power: float,
    turbulence_power: float,
    max_delay: int,
    vecnorm_path: str | None = None,
):
    env_fns = [
        make_single_env(
            seed + i,
            noise_std,
            wind_power,
            turbulence_power,
            max_delay,
        )
        for i in range(n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns)

    if vecnorm_path:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = True
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,   # LunarLander
            clip_obs=10.0,
            gamma=0.99,
        )

    return vec_env


# ==================================================
# LEARNING RATE SCHEDULE
# ==================================================

def linear_schedule(initial_lr):
    def schedule(progress):
        return initial * (0.2 + 0.8 * progress)
    return schedule


# ==================================================
# TRAINING PHASE
# ==================================================

def train_phase(
    name: str,
    total_timesteps: int,
    env_kwargs: dict,
    model: PPO | None = None,
    vecnorm_path: str | None = None,
):
    print(f"\n===== {name} =====")

    env = make_vec_env_robust(
        **env_kwargs,
        vecnorm_path=vecnorm_path,
    )

    if model is None:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(3e-4),
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.003,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[512, 512]),
                activation_fn=torch.nn.ReLU,
                ortho_init=True,
            ),
            seed=42,
            verbose=1,
        )
    else:
        model.set_env(env)

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=10,
    )

    return model, env


# ==================================================
# MAIN (WINDOWS SAFE)
# ==================================================

if __name__ == "__main__":

    logger = configure("logs/ppo_lander_robust/", ["tensorboard"])

    N_ENVS = 8

    # -------------------------
    # PHASE 1 – BASELINE
    # -------------------------
    model, env = train_phase(
        "PHASE 1 – Sans perturbations",
        total_timesteps=1_000_000,
        env_kwargs=dict(
            n_envs=N_ENVS,
            seed=42,
            noise_std=0.0,
            wind_power=0.0,
            turbulence_power=0.0,
            max_delay=0,
        ),
    )
    env.save("models/vecnorm_phase1.pkl")

    # -------------------------
    # PHASE 2 – PERTURBATIONS MODÉRÉES
    # -------------------------
    model, env = train_phase(
        "PHASE 2 – Vent + bruit",
        total_timesteps=1_500_000,
        env_kwargs=dict(
            n_envs=N_ENVS,
            seed=100,
            noise_std=0.04,
            wind_power=10.0,
            turbulence_power=1.0,
            max_delay=1,
        ),
        model=model,
        vecnorm_path="models/vecnorm_phase1.pkl",
    )
    env.save("models/vecnorm_phase2.pkl")

    # -------------------------
    # PHASE 3 – ROBUST FINAL
    # -------------------------
    model, env = train_phase(
        "PHASE 3 – Setup robuste final",
        total_timesteps=2_000_000,
        env_kwargs=dict(
            n_envs=N_ENVS,
            seed=200,
            noise_std=0.08,
            wind_power=18.0,
            turbulence_power=1.8,
            max_delay=3,
        ),
        model=model,
        vecnorm_path="models/vecnorm_phase2.pkl",
    )
    env.save("models/vecnorm_phase3.pkl")

    model.save("models/ppo_lunar_lander_robust_final")

    print("\nENTRAÎNEMENT TERMINÉ")
