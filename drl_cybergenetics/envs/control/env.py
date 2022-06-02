#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np


class Env(abc.ABC):
    """An environment with arbitrary behind-the-scenes dynamics, which can be
    partially or fully observed.

    OpenAI Gym style API methods:

        reset
        step
        render
        close
        seed

    """

    def __init__(self) -> None:
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: typing.Union[int, np.ndarray]):
        raise NotImplementedError

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError

    def close(self) -> None:
        pass

    def seed(self, seed: typing.Optional[int] = None) -> None:
        self._rng.seed(seed)

    @property
    def unwrapped(self):
        return self


class Wrapper(Env):
    """A wrapper of an environment to allow modular transformation."""

    def __init__(self, env: Env) -> None:
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def step(self, action: typing.Union[int, np.ndarray], **kwargs):
        return self.env.step(action, **kwargs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self) -> None:
        return self.env.close()

    def seed(self, seed: typing.Optional[int] = None) -> None:
        return self.env.seed(seed=seed)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped


class ObservationWrapper(Wrapper):

    def reset(self):
        return self.observe(self.env.reset())

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.observe(observation), reward, done, info

    def observe(self, observation):
        raise NotImplementedError
