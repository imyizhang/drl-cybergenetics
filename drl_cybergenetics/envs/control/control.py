#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from typing import (
    Any,
    Optional,
    Tuple,
    NamedTuple,
)
import contextlib
import copy

import numpy as np

from .env import ObsType, ActType, Env
from .spaces import Space


_Fields = (
    'time',
    'control',
    'physical_action',
    'state',
    'timestep',
    'action',
    'reference',
    'observation',
    'reward',
)


class Timestep(NamedTuple):

    fields = _Fields

    time: float
    control: Any
    physical_action: Any
    state: ObsType
    timestep: int
    action: ActType
    reference: ObsType
    observation: ObsType
    reward: float

    def to_dict(self) -> dict:
        return self._asdict()


class Trajectory(Timestep):
    pass


class Buffer(object):

    def __init__(self) -> None:
        self._data = []

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, attr: str):
        if attr.startwith('_'):
            raise AttributeError
        return getattr(self.trajectory, attr)

    def __copy__(self):
        # TODO: handle copy
        raise NotImplementedError

    def copy(self):
        instance = self.__class__.__new__(self.__class__)
        instance.__dict__['_data'] = copy.deepcopy(self.__dict__['_data'])
        return instance

    def empty(self) -> bool:
        return len(self._data) == 0

    def clear(self) -> None:
        self._data.clear()

    def push(self, *timestep) -> None:
        self._data.append(Timestep(*timestep))

    @property
    def timestep(self) -> Timestep:
        if len(self._data) == 0:
            return None
        return self._data[-1]

    @property
    def trajectory(self) -> Trajectory:
        if len(self._data) == 0:
            return None
        return Trajectory(*zip(*self._data))


class Physics(abc.ABC):

    _time = 0.0
    _control = None
    _physical_action = None
    _state = None

    def __init__(self) -> None:
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def dynamics(self, time: float, state: ObsType, physical_action: Any):
        raise NotImplementedError

    def set_control(self, action: Any) -> None:
        self._control = action
        self._physical_action = self._control

    @contextlib.contextmanager
    def reset_context(self):
        try:
            self.reset()
        finally:
            yield self

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    def time(self) -> float:
        return self._time

    def control(self) -> Any:
        return self._control

    def physical_action(self) -> Any:
        return self._physical_action

    def state(self) -> ObsType:
        return self._state

    def terminal(self) -> bool:
        return False

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)


class Task(abc.ABC):

    _timestep = 0
    _reference = None
    _observation = None
    _reward = None

    def __init__(self) -> None:
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def action_space(self, physics: Physics) -> Space:
        raise NotImplementedError

    @abc.abstractmethod
    def observation_space(self, physics: Physics) -> Space:
        raise NotImplementedError

    def reset(self, physics: Physics) -> None:
        self._timestep = 0
        self._reference = None
        self._observation = None
        self._reward = None

    def before_step(self, action: ActType, physics: Physics) -> None:
        physics.set_control(action)

    def step(self, physics: Physics) -> None:
        physics.step()
        self._timestep += 1

    def after_step(self, physics: Physics) -> None:
        pass

    def timestep(self) -> int:
        return self._timestep

    def reference(self, physcis: Physics) -> ObsType:
        raise NotImplementedError

    def observation(self, physics: Physics) -> ObsType:
        raise NotImplementedError

    def reward(self, physics: Physics) -> float:
        raise NotImplementedError

    def done(self, physics: Physics) -> bool:
        return physics.terminal()

    def info(self, physics: Physics) -> dict:
        return {}

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)


class Environment(Env):

    _buffer = Buffer()

    def __init__(self, physics: Physics, task: Task):
        super().__init__()
        self._physics = physics
        self._task = task
        self.action_space = self._task.action_space(self._physics)
        self.observation_space = self._task.observation_space(self._physics)

    @property
    def buffer(self) -> Buffer:
        return self._buffer

    @property
    def physics(self) -> Physics:
        return self._physics

    @property
    def task(self) -> Task:
        return self._task

    def reset(self) -> ObsType:
        self._buffer.clear()
        with self._physics.reset_context():
            self._task.reset(self._physics)
        time = self._physics.time()
        control = None
        physical_action = None
        state = self._physics.state()
        timestep = self._task.timestep()
        action = None
        reference = self._task.reference(self._physics)
        observation = self._task.observation(self._physics)
        reward = None
        self._buffer.append(
            time, control, physical_action, state,
            timestep, action, reference, observation, reward,
        )
        return observation

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        if self._buffer.empty():
            raise RuntimeError
        self._task.before_step(action, self._physics)
        self._task.step(self._physics)
        self._task.after_step(self._physics)
        time = self._physics.time()
        control = self._physics.control()
        physical_action = self._physics.physical_action()
        state = self._physics.state()
        timestep = self._task.timestep()
        reference = self._task.reference(self._physics)
        observation = self._task.observation(self._physics)
        reward = self._task.reward(self._physics)
        done = self._task.done(self._physics)
        info = self._task.info(self._physics)
        self._buffer.append(
            time, control, physical_action, state,
            timestep, action, reference, observation, reward,
        )
        return observation, reward, done, info

    def close(self) -> None:
        self._buffer = Buffer()

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)
        self._physics.seed(seed)
        self._task.seed(seed)
        self.action_space.seed(seed)
