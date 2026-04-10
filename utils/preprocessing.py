import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class GrayScaleObservation(gym.ObservationWrapper):
    """Convert observations to grayscale."""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs):
        # Convert RGB to grayscale using luminosity method
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return obs


class ResizeObservation(gym.ObservationWrapper):
    """Resize observations to 84x84."""
    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return obs


class FrameStack(gym.ObservationWrapper):
    """Stack the last n frames."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        obs_shape = (n_frames,) + env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the frame buffer with the initial frame
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(self.frames, dtype=np.uint8)


class ScaleObservation(gym.ObservationWrapper):
    """Scale observations to [0, 1] range."""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


def make_atari_env(env_name='ALE/Pong-v5', seed=42, render_mode=None):
    """
    Create an Atari environment with standard DQN preprocessing.
    
    Args:
        env_name: Name of the Atari game
        seed: Random seed
        render_mode: 'human' for visualization, None for training
    
    Returns:
        Preprocessed environment
    """
    import ale_py
    gym.register_envs(ale_py)
    
    # Create base environment
    env = gym.make(env_name, render_mode=render_mode)
    env.reset(seed=seed)
    
    # Apply preprocessing wrappers
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, n_frames=4)
    env = ScaleObservation(env)
    
    return env