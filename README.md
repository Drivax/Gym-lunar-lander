# Gym LunarLander: Robust Reinforcement Learning with Delay, Noise, and Wind

## Project Objective

This project studies how to train a LunarLander policy that still works when conditions are no longer ideal. I started from the standard `LunarLander-v3` environment and then introduced realistic disturbances: delayed actions, noisy observations, and wind.

The main goal is simple: learn a control policy that stays stable and lands safely even when the input and dynamics are imperfect. The full pipeline is built around PPO, multi-phase training, and careful evaluation with reproducible settings.

## Dataset

This project does not use a fixed offline dataset. Data is generated online by interacting with the Gymnasium environment.

Source:
- Environment: Gymnasium `LunarLander-v3` (Box2D dynamics)

Size and scale:
- Observation vector: 8 state features in the base environment
- Augmented observation vector: 9 features when delayed action is appended
- Training rollouts: generated from 8 parallel environments (`n_envs = 8`)
- Baseline training budget: `3,000,000` timesteps
- Fine-tuning budget: `1,000,000` timesteps
- Evaluation sets used in this repository:
- `advanced_eval/metrics.json` built from `100` episodes
- `diagnostic/diagnostic.json` built from `100` episodes

Key features used by the model:
- Position: `x`, `y`
- Velocity: `v_x`, `v_y`
- Angle and angular velocity: `theta`, `omega`
- Contact flags: left leg, right leg
- Last executed action (added by wrapper when delay-aware mode is enabled)

## Methodology

### 1. Environment design for robustness

I kept the base LunarLander dynamics, then added controlled perturbations:
- Action delay with a random delay up to 3 steps
- Observation noise with Gaussian perturbation (`sigma = 0.08`)
- Wind and turbulence perturbations

The delay-aware wrapper appends the last executed action to the observation. This helps the policy infer recent actuator behavior and reduces ambiguity caused by latency.

### 2. PPO training strategy

The policy is trained with Stable-Baselines3 PPO and a shared MLP (`[256, 256]`, `Tanh`).

Training is split into phases:
- Phase 1 (baseline): clean environment, no perturbations
- Phase 2 (fine-tuning): light wind, lower learning rate, reduced entropy coefficient
- Phase 3 (robustness): heavier perturbations (wind + delay + noise), with evaluation and diagnostics

Normalization is handled with `VecNormalize` for observation and reward stability during learning.

### 3. Evaluation and diagnostics

Evaluation is deterministic by default for clear comparison across runs. I compute both standard outcomes (reward, success-like thresholds, trajectory length) and policy-level indicators (entropy, KL trend, instability).

The repository also stores diagnostic signals such as oscillation index, control energy, and wind generalization checks. This makes it easier to inspect *how* the policy behaves, not only final reward.

## Key Equations

### PPO clipped objective

$$
L_{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

This is the core PPO update. The ratio term compares new and old policy probabilities, and clipping prevents updates that are too large, which keeps training stable.

### Importance ratio

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

If this ratio moves too far from `1`, the policy changed too much in one step. PPO clips this effect to avoid destructive jumps.

### Generalized Advantage Estimation (GAE)

$$
\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}^{V}, \quad
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GAE blends short-term and long-term credit assignment. It lowers variance compared to plain returns, while keeping useful learning signal.

### Observation noise model

$$
\hat{o}_t = o_t + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\sigma^2 I)
$$

This simulates sensor uncertainty. The policy must act correctly even when each state readout is slightly corrupted.

### Delay-aware observation

$$
\tilde{o}_t = [o_t; a_{t-d}] \in \mathbb{R}^{9}
$$

Appending the delayed action gives the policy context about recent control commands. This is a practical way to reduce partial observability introduced by latency.

## Evaluation

Values below are copied from repository outputs and kept exact.

From `advanced_eval/metrics.json` (`100` episodes):
- `reward_mean`: `275.4732666015625`
- `reward_std`: `42.905555725097656`
- `entropy_mean`: `0.4174534260646704`
- `entropy_std`: `0.3240422878998123`
- `kl_mean`: `0.0`
- `instability`: `42.905555725097656`
- `catastrophic_forgetting`: `0.943634033203125`
- `landing_distribution.UNSTABLE`: `100`

From `diagnostic/diagnostic.json` (`100` episodes):
- `mean_reward`: `282.94891357421875`
- `reward_std`: `19.46548080444336`
- `success_rate`: `0.0`
- `surcontrol`: `true`
- `oscillates`: `true`
- `optimal_policy`: `true`
- `generalizes_wind`: `true`
- `overfit`: `false`
- `bangbang_control`: `true`
- `control_energy`: `614.07`
- `oscillation_index`: `7.97804314839113`
- `wind_rewards`: `[276.98602294921875, 277.93194580078125, 278.36248779296875, 278.58740234375]`

## Repository Structure

```text
Gym-lunar-lander/
├── train.py                     # Training pipeline and analysis helpers
├── evaluation.py                # Deterministic policy evaluation script
├── test_env.ipynb              # Notebook for quick experiments and checks
├── requirements.txt            # Python dependencies
├── envs/
│   ├── __init__.py
│   └── wrappers.py             # Action delay and observation noise wrappers
├── models/                     # Saved PPO models and VecNormalize stats
├── logs/                       # TensorBoard logs and evaluation archives
├── advanced_eval/
│   └── metrics.json            # Advanced evaluation metrics
├── diagnostic/
│   └── diagnostic.json         # Policy behavior diagnostics
├── gifs/                       # Training / policy GIF outputs
├── gifs_eval/                  # Evaluation GIF outputs
└── pics/                       # Optional generated plots
```

## Installation and Execution

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train baseline model

```bash
python train.py --phase baseline
```

### 3. Fine-tune from baseline checkpoint

```bash
python train.py --phase finetune
```

### 4. Run deterministic evaluation

```bash
python evaluation.py --model-path models/ppo_final.zip --vecnorm-path models/vecnorm_baseline.pkl --episodes 50
```

### 5. Open notebook for manual checks

```bash
jupyter notebook test_env.ipynb
```
