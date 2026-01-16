# Gym-lunar-lander
Reinforcement learning for LunarLander with action delay, noisy observations, and environmental perturbations.

##  Environment Modifications

## 1. Noisy Observations
Gaussian noise is injected into the state vector to simulate sensor uncertainty.

```text
obs_noisy = obs + N(0, σ²)
```

## 2. Action Delay (Delay-Aware)

The last action is explicitly appended to the observation, allowing the policy to infer delayed dynamics.

```text
observation = [env_state, last_action]
```
## PPO architecture :
```text
Input (10)
 → Dense(64) → ReLU
 → Dense(64) → ReLU
 → Policy Head (actions)
 → Value Head (V(s))
```
