#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from typing import (
    TypeVar,
    Generic,
    Any,
    Optional,
    Tuple,
)

import numpy as np

from .spaces import Space


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


class Env(Generic[ObsType, ActType]):
    """Environment with arbitrary behind-the-scenes dynamics, which can be
    partially or fully observed.
    """

    observation_space: Space[ObsType]
    action_space: Space[ActType]

    def __init__(self) -> None:
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def reset(self) -> ObsType:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        raise NotImplementedError

    @abc.abstractmethod
    def render(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)

    @property
    def unwrapped(self):
        return self


class Wrapper(Env[ObsType, ActType]):
    """Wrapper of an environment to allow modular transformation."""

    def __init__(self, env: Env) -> None:
        self.env = env
        self._observation_space = None
        self._action_space = None

    def __getattr__(self, attr: str):
        if attr.startwith('_'):
            raise AttributeError
        return getattr(self.env, attr)

    @property
    def observation_space(self) -> Space:
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: Space):
        self._observation_space = space

    @property
    def action_space(self) -> Space[ActType]:
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: Space):
        self._action_space = space

    def reset(self) -> ObsType:
        return self.env.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return self.env.step(action)

    def render(self, **kwargs) -> Any:
        return self.env.render(**kwargs)

    def close(self) -> None:
        return self.env.close()

    def seed(self, seed: Optional[int] = None) -> None:
        return self.env.seed(seed=seed)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped


class ObservationWrapper(Wrapper):
    """Wrapper that can modify observation via `observation()` method."""

    def reset(self) -> ObsType:
        return self.observation(self.env.reset())

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: ObsType) -> ObsType:
        raise NotImplementedError


class ActionWrapper(Wrapper):
    """Wrapper that can modify action via `action()` method."""

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return self.env.step(self.action(action))

    def action(self, action: ActType) -> ActType:
        raise NotImplementedError
