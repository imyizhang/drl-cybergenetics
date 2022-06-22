#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Union
import pathlib

import numpy as np
import torch
import imageio

from .env import (
    ObsType,
    ActType,
    Env,
    Wrapper,
    ObservationWrapper,
    ActionWrapper,
)
from .spaces import Discrete, Box


class AsTensor(Wrapper):
    """Wrapper that transforms data types to PyTorch tensors."""

    def __init__(
        self,
        env: Env,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(env)
        self.dtype = dtype
        self.device = device

    @property
    def state_dim(self) -> int:
        return self.observation_space.shape[0]

    @property
    def action_dim(self) -> int:
        if isinstance(self.action_space, Discrete):
            return self.action_space.n
        else:
            return self.action_space.shape[0]

    @property
    def action_sample(self) -> torch.Tensor:
        action = self.action_space.sample()
        return self.as_tensor(action, self.dtype, self.device)

    def reset(self) -> torch.Tensor:
        observation = self.env.reset()
        return self.as_tensor(observation, self.dtype, self.device)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if isinstance(self.action_space, Discrete):
            action = action.cpu().detach().item()
        else:
            action = action.view(-1).cpu().detach().numpy()
        observation, reward, done, info = self.env.step(action)
        observation = self.as_tensor(observation, self.dtype, self.device)
        reward = self.as_tensor(reward, self.dtype, self.device)
        done = self.as_tensor(done, self.dtype, self.device)
        return observation, reward, done, info

    @staticmethod
    def as_tensor(data, dtype, device):
        return torch.as_tensor(data, dtype=dtype, device=device).view(1, -1)


class LimitTimestep(Wrapper):
    """Wrapper that limits timesteps per episode."""

    def __init__(self, env: Env, max_episode_steps: int) -> None:
        super().__init__(env)
        self.max_episode_steps = max_episode_steps

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        observation, reward, done, info = super().step(action)
        if len(self.buffer) > self.max_episode_steps:
            info['truncated'] = not done
            done = True
        return observation, reward, done, info


class TrackEpisode(Wrapper):
    """Wrapper that keeps track of cumulative reward and episode length."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.episode_return = None
        self.episode_length = None

    def reset(self) -> ObsType:
        observation = super().reset()
        self.episode_return = 0.0
        self.episode_length = 0
        return observation

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        observation, reward, done, info = super().step(action)
        self.episode_return += reward
        self.episode_length += 1
        info['episode_return'] = self.episode_return
        info['episode_length'] = self.episode_length
        return observation, reward, done, info


class RecordEpisode(Wrapper):
    """Wrapper that records video of episode."""

    def __init__(self, env: Env, path: Union[str, pathlib.Path], **render_kwargs) -> None:
        super().__init__(env)
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.render_kwargs = render_kwargs
        self._frames = []

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        observation, reward, done, info = super().step(action)
        fig = super().render(**self.render_kwargs)
        frame = self.path.joinpath(f'fig{self.buffer.curr_timestep.timestep}.png')
        fig.savefig(frame)
        self._frames.append(frame)
        if done:
            with imageio.get_writer('episode.gif', mode='I') as writer:
                for frame in self._frames:
                    writer.append_data(imageio.imread(frame))
        return observation, reward, done, info


class FullObservation(ObservationWrapper):
    """Wrapper that observes physical internal state."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.observation_space = Box(
            low=self.physics.state_min,
            high=self.physics.state_max,
            dtype=self.physics.dtype,
        )

    def observation(self, observation: ObsType) -> ObsType:
        return self.buffer.curr_timestep.state


class TimeAwareObservation(ObservationWrapper):
    """Wrapper that augments the observation with current time."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.observation_space = Box(
            low=np.append(self.observation_space.low, 0.0),
            high=np.append(self.observation_space.high, np.inf),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: ObsType) -> ObsType:
        return np.append(observation, self.buffer.curr_timestep.time).astype(self.observation_space.dtype)


class TimestepAwareObservation(ObservationWrapper):
    """Wrapper that augments the observation with current timestep."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.observation_space = Box(
            low=np.append(self.observation_space.low, 0.0),
            high=np.append(self.observation_space.high, np.inf),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: ObsType) -> ObsType:
        return np.append(observation, self.buffer.curr_timestep.timestep).astype(self.observation_space.dtype)


class ReferenceAwareObservation(ObservationWrapper):
    """Wrapper that augments the observation with current reference."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.observation_space = Box(
            low=np.append(self.observation_space.low, 0.0),
            high=np.append(self.observation_space.high, np.inf),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: ObsType) -> ObsType:
        return np.append(observation, self.buffer.curr_timestep.reference).astype(self.observation_space.dtype)


class ToleranceAwareObservation(ObservationWrapper):
    """Wrapper that augments the observation with current tolerance."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.observation_space = Box(
            low=np.append(self.observation_space.low, 0.0),
            high=np.append(self.observation_space.high, np.inf),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: ObsType) -> ObsType:
        reference = self.buffer.curr_timestep.reference
        tolerance = self.task.tolerance
        _in_tolerance = self.in_tolerance(observation, reference, tolerance)
        return np.append(observation, _in_tolerance).astype(self.observation_space.dtype)

    @staticmethod
    def in_tolerance(achieved, desired, tolerance):
        return float(np.abs(achieved - desired) / desired < tolerance)


class ActionAwareObservation(ObservationWrapper):
    """Wrapper that augments the observation with current action."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.observation_space = Box(
            low=np.append(self.observation_space.low, 0.0),
            high=np.append(self.observation_space.high, np.inf),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: ObsType) -> ObsType:
        return np.append(observation, self.buffer.curr_timestep.action).astype(self.observation_space.dtype)


class RescaleAction(ActionWrapper):
    """Wrapper that affinely rescales continuous action space."""

    def __init__(
        self,
        env: Env,
        action_min: Union[float, np.ndarray] = 0.0,
        action_max: Union[float, np.ndarray] = 1.0,
    ) -> None:
        super().__init__(env)
        self._from_range = (self.action_space.low, self.action_space.high)
        self.action_space = Box(
            low=action_min,
            high=action_max,
            shape=(1,),
            dtype=self.action_space.dtype,
        )
        self._to_range = (self.action_space.low, self.action_space.high)

    def action(self, action: ActType) -> ActType:
        raise self.rescale(action, self._from_range, self._to_range)

    @staticmethod
    def rescale(data, from_range, to_range):
        _low, _high = from_range
        low, high = to_range
        return low + (data - _low) * (high - low) / (_high - _low)
