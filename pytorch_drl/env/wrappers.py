#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

from .crn import Wrapper, make, spaces


class CRNEnv(Wrapper):
    """A wrapper, which transforms numpy array to torch tensor."""

    def __init__(
        self,
        env,
        partial_observation=True,
        time_aware=False,
        tolerance_aware=False,
        max_timesteps=None,
        device=None,
        dtype=torch.float32,
        **kwargs,
    ):
        env = make(env, **kwargs) if isinstance(env, str) else env
        if partial_observation:
            env = PartialObservation(env)
        if time_aware:
            env = TimeAwareObservation(env)
        if tolerance_aware:
            env = ToleranceAwareObservation(env)
        if max_timesteps is not None:
            env = LimitedTimestep(env, max_timesteps)
        env = ToTensor(env, device, dtype)
        super().__init__(env)


class ToTensor(Wrapper):
    """A wrapper, which transforms numpy array to torch tensor."""

    def __init__(self, env, device=None, dtype=torch.float32):
        super().__init__(env)
        self.device = device
        self.dtype = dtype

    @property
    def state_dim(self):
        return self.env.observation_space.shape[0]

    @property
    def action_dim(self):
        if isinstance(self.env.action_space, spaces.Discrete):
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    @property
    def action_sample(self):
        action = self.env.action_space.sample()
        return torch.as_tensor(action, dtype=self.dtype, device=self.device).view(1, -1)

    def reset(self):
        state = self.env.reset()
        return torch.as_tensor(state, dtype=self.dtype, device=self.device).view(1, -1)

    def step(self, action, **kwargs):
        if isinstance(self.env.action_space, spaces.Discrete):
            action = action.cpu().detach().item()  # int
        else:
            action = action.view(-1).cpu().detach().numpy()  # np.ndarray
        state, reward, done, info = self.env.step(action, **kwargs)
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device).view(1, -1)
        reward = torch.as_tensor(reward, dtype=self.dtype, device=self.device).view(1, 1)
        done = torch.as_tensor(done, dtype=torch.bool, device=self.device).view(1, 1)
        return state, reward, done, info


class LimitedTimestep(Wrapper):
    """A wrapper, which limits timesteps."""

    def __init__(self, env, max_timesteps):
        super().__init__(env)
        self.max_timesteps = max_timesteps

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        if self.env._cache.steps_done >= self.max_timesteps:
            done = True
        return observation, reward, done, info


class ObservationWrapper(Wrapper):

    def reset(self):
        return self.observe(self.env.reset())

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.observe(observation), reward, done, info

    def observe(self, observation):
        raise NotImplementedError


class PartialObservation(ObservationWrapper):
    """A wrapper, which partially observes states."""

    def __init__(self, env):
        super().__init__(env)
        self.env.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=(1,),
            dtype=self.env.observation_space.dtype,
        )

    def observe(self, observation):
        return self.env._cache.observations[self.env._cache.steps_done]


class TimeAwareObservation(ObservationWrapper):
    """A wrapper, which augments the observation with current time."""

    def __init__(self, env):
        super().__init__(env)
        self.env.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=(self.env.observation_space.shape[0] + 1,),
            dtype=self.env.observation_space.dtype,
        )

    def observe(self, observation):
        return np.append(observation, self.env._cache.t[self.env._cache.steps_done])


class ToleranceAwareObservation(ObservationWrapper):
    """A wrapper, which augments the observation with current tolerance."""

    def __init__(self, env):
        super().__init__(env)
        self.env.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=(self.env.observation_space.shape[0] + 1,),
            dtype=self.env.observation_space.dtype,
        )

    def observe(self, observation):
        achieved = self.env._cache.observations[self.env._cache.steps_done]
        desired, _ = self.env.ref_trajectory(self.env._cache.t[self.env._cache.steps_done])
        achieved, desired = float(achieved[0]), float(desired[0])
        # within tolerance: 1
        if abs(achieved - desired) / desired < self.ref_trajectory.tolerance:
            state_in_tolerance = np.array([1], dtype=self.env.observation_space.dtype)
        # below tolerance: 0
        elif achieved < desired :
            state_in_tolerance = np.array([0], dtype=self.env.observation_space.dtype)
        # above tolerance: 2
        else:
            state_in_tolerance = np.array([2], dtype=self.env.observation_space.dtype)
        return np.append(observation, state_in_tolerance)
