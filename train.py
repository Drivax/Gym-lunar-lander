import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.wrappers import NoisyObservations, ActionDelayAware

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
        make_single_env(seed + i, noise_std, wind_power, turbulence_power, max_delay)
        for i in range(n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns)

    if vecnorm_path:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=50.0,
            gamma=0.99,
        )

    return vec_env


def linear_schedule(initial_lr: float, final_lr_ratio: float = 0.2):
    def schedule(progress_remaining: float) -> float:
        return initial_lr * (final_lr_ratio + (1 - final_lr_ratio) * progress_remaining)
    return schedule


def train_phase(
    name: str,
    total_timesteps: int,
    env_kwargs: dict,
    model: PPO | None = None,
    vecnorm_path: str | None = None,
    eval_freq: int = 50000,
    save_freq: int = 250000,
):
    print(f"\n===== {name} – {total_timesteps:,} timesteps =====")

    train_env = make_vec_env_robust(**env_kwargs, vecnorm_path=vecnorm_path)

    eval_env = make_vec_env_robust(
        n_envs=1,
        seed=env_kwargs["seed"] + 10000,
        **{k: v for k, v in env_kwargs.items() if k != "n_envs" and k != "seed"},
        vecnorm_path=None,
    )
    eval_env = VecNormalize.load(vecnorm_path or "dummy", eval_env) if vecnorm_path else VecNormalize(eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    callbacks = []

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/best_{name.lower().replace(' ', '_')}",
        log_path=f"logs/{name.lower().replace(' ', '_')}",
        eval_freq=eval_freq // env_kwargs["n_envs"],
        deterministic=True,
        render=False,
        n_eval_episodes=15,
    )
    callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // env_kwargs["n_envs"],
        save_path=f"models/checkpoints_{name.lower().replace(' ', '_')}",
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    if model is None:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(3e-4),
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[512, 256, 128]),
                activation_fn=torch.nn.Tanh,
                ortho_init=True,
            ),
            tensorboard_log="logs/ppo_lander_robust/",
            seed=42,
            verbose=1,
        )
    else:
        model.set_env(train_env)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=5,
    )

    vecnorm_final_path = f"models/vecnorm_{name.lower().replace(' ', '_')}.pkl"
    train_env.save(vecnorm_final_path)

    return model, train_env, vecnorm_final_path


if __name__ == "__main__":

    logger = configure("logs/ppo_lander_robust/", ["tensorboard"])

    N_ENVS = 12

    # PHASE 1 – BASELINE (sans perturbations)
    model, env1, vecnorm1 = train_phase(
        "PHASE 1 – Baseline",
        total_timesteps=1_200_000,
        env_kwargs=dict(
            n_envs=N_ENVS,
            seed=42,
            noise_std=0.0,
            wind_power=0.0,
            turbulence_power=0.0,
            max_delay=0,
        ),
    )

    # PHASE 2 – Perturbations modérées
    model, env2, vecnorm2 = train_phase(
        "PHASE 2 – Perturbations modérées",
        total_timesteps=1_800_000,
        env_kwargs=dict(
            n_envs=N_ENVS,
            seed=100,
            noise_std=0.04,
            wind_power=10.0,
            turbulence_power=1.0,
            max_delay=1,
        ),
        model=model,
        vecnorm_path=vecnorm1,
    )

    # PHASE 3 – Perturbations fortes (final robuste)
    model, env3, vecnorm3 = train_phase(
        "PHASE 3 – Robust Final",
        total_timesteps=3_000_000,
        env_kwargs=dict(
            n_envs=N_ENVS,
            seed=200,
            noise_std=0.08,
            wind_power=18.0,
            turbulence_power=1.8,
            max_delay=3,
        ),
        model=model,
        vecnorm_path=vecnorm2,
    )

    model.save("models/ppo_lunar_lander_robust_final")

    print("\nENTRAÎNEMENT TERMINÉ")
    print("→ Modèle final : models/ppo_lunar_lander_robust_final.zip")
    print("→ VecNormalize final : models/vecnorm_phase_3_robust.pkl")
    print("→ Logs TensorBoard : tensorboard --logdir logs/ppo_lander_robust/")