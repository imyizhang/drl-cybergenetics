#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing
import contextlib

import numpy as np

from .env import Env


class Environment(Env):

    def __init__(self, physics, task):
        self._physics = physics
        self._task = task
        self.action_space = self._task.action_spec(self._physics)
        self.observation_space = self._task.observation_spec(self._physics)

    @property
    def physics(self):
        return self._physics

    @property
    def task(self):
        return self._task

    def reset(self):
        with self._physics.reset_context():
            self._task.reset(self._physics)
        observation = self._task.observation(self._physics)
        return observation

    def step(self, action):
        self._task.before_step(self._physics)
        self._task.step(self._physics)
        self._task.after_step(self._physics)
        observation = self._task.observation(self._physics)
        reward = self._task.reward(self._physics)
        done = self._task.done(self._physics)
        info = self._task.info(self._physics)
        return observation, reward, done, info


class Physics(abc.ABC):

    def __init__(self) -> None:
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def dynamics(self, t, state, action, noise):
        raise NotImplementedError

    def set_control(self, action):
        raise NotImplementedError

    def state(self):
        raise NotImplementedError

    def time(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def reset_context(self):
        try:
            self.reset()
        finally:
            yield self

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    def seed(self, seed: typing.Optional[int] = None) -> None:
        self._rng.seed(seed)


class Task(abc.ABC):

    @abc.abstractmethod
    def action_spec(self, physics):
        raise NotImplementedError

    @abc.abstractmethod
    def observation_spec(self, physics):
        raise NotImplementedError

    def timestep_spec(self, physics):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, physics):
        raise NotImplementedError

    @abc.abstractmethod
    def before_step(self, action, physics):
        raise NotImplementedError

    def step(self, physics):
        return physics.step()

    def after_step(self, physics):
        raise NotImplementedError

    @abc.abstractmethod
    def observation(self, physics):
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self, physics):
        raise NotImplementedError

    def done(self, physics):
        raise NotImplementedError

    def info(self, physics):
        raise NotImplementedError
