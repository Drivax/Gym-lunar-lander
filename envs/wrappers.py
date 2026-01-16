import gymnasium as gym
import numpy as np
from collections import deque


class NoisyObservations(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.08):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return (obs + noise).astype(np.float32)


class ActionDelayAware(gym.Wrapper):
    """
    Action delay + last_action incluse dans l'observation.
    Compatible Discrete et Box.
    """

    def __init__(self, env, max_delay=3):
        super().__init__(env)
        self.max_delay = max_delay
        self.delay_queue = deque()
        self.current_delay = 0

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = 1
        else:
            self.action_dim = int(np.prod(env.action_space.shape))

        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        low = np.concatenate([env.observation_space.low,
                              -np.ones(self.action_dim)])
        high = np.concatenate([env.observation_space.high,
                               np.ones(self.action_dim)])

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = env.action_space

    def reset(self, **kwargs):
        self.delay_queue.clear()
        self.current_delay = np.random.randint(0, self.max_delay + 1)
        self.last_action[:] = 0.0

        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self.last_action]), info

    def step(self, action):
        # Discrete-safe
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = int(action)
            self.last_action[0] = action
        else:
            action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
            self.last_action = action.copy()

        self.delay_queue.append(action)

        if len(self.delay_queue) > self.current_delay:
            exec_action = self.delay_queue.popleft()
        else:
            exec_action = 0 if isinstance(self.action_space, gym.spaces.Discrete) \
                else np.zeros(self.action_dim, dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.concatenate([obs, self.last_action])

        return obs, reward, terminated, truncated, info

class ActionDelayAwareEval(gym.Wrapper):
    """
    Wrapper d'évaluation :
    - applique un délai d'action aléatoire
    - ajoute toujours last_action à l'observation
    - observation = [obs_env, last_action]
    """

    def __init__(self, env, max_delay=0):
        super().__init__(env)

        self.max_delay = max_delay
        self.delay_queue = deque()
        self.current_delay = 0

        # Dimension action
        self.action_dim = int(np.prod(env.action_space.shape))
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        # Observation space étendue
        low = np.concatenate([
            env.observation_space.low,
            -np.ones(self.action_dim, dtype=np.float32)
        ])
        high = np.concatenate([
            env.observation_space.high,
            np.ones(self.action_dim, dtype=np.float32)
        ])

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self.delay_queue.clear()

        self.current_delay = (
            np.random.randint(0, self.max_delay + 1)
            if self.max_delay > 0 else 0
        )

        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)

        obs = np.concatenate([obs, self.last_action])
        return obs, info

    def step(self, action):
        # Sécurisation stricte des dimensions
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.last_action = action.copy()

        # Application du délai
        if self.max_delay > 0:
            self.delay_queue.append(action)
            if len(self.delay_queue) > self.current_delay:
                exec_action = self.delay_queue.popleft()
            else:
                exec_action = np.zeros(self.action_dim, dtype=np.float32)
        else:
            exec_action = action

        obs, reward, terminated, truncated, info = self.env.step(exec_action)

        obs = np.asarray(obs, dtype=np.float32)
        obs = np.concatenate([obs, self.last_action])

        return obs, reward, terminated, truncated, info