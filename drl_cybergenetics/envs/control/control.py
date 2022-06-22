#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from typing import Any, Optional, Tuple, NamedTuple
import contextlib
import copy

import numpy as np

from .env import ObsType, ActType, Env
from .spaces import Space


_Fields = (
    'time',
    'timestep',
    'control',
    'physical_control',
    'state',
    'action',
    'reference',
    'observation',
    'reward',
)


class Timestep(NamedTuple):

    fields = _Fields

    time: float
    timestep: int
    control: Any
    physical_control: Any
    state: ObsType
    action: ActType
    reference: ObsType
    observation: ObsType
    reward: float

    def as_dict(self) -> dict:
        return self._asdict()


class Trajectory(Timestep):
    pass


class Buffer(object):

    def __init__(self) -> None:
        self._data = []

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, attr: str):
        if attr.startswith('_'):
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
    def curr_timestep(self) -> Timestep:
        if len(self._data) == 0:
            return None
        return self._data[-1]

    @property
    def trajectory(self) -> Trajectory:
        if len(self._data) == 0:
            return None
        return Trajectory(*zip(*self._data))


class Physics(abc.ABC):
    """Simulates a controlled dynamical system from physics."""

    _time = 0.0
    _timestep = 0
    _state = None
    _control = None
    _physical_control = None

    def __init__(self) -> None:
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def dynamics(self, time: float, state: ObsType, control: Any) -> ObsType:
        raise NotImplementedError

    def set_control(self, control: Any) -> None:
        self._control = control
        self._physical_control = self._control

    @contextlib.contextmanager
    def reset_context(self):
        try:
            self.reset()
        except PhysicsError:
            pass
        finally:
            yield self

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, sampling_rate: float) -> None:
        raise NotImplementedError

    def time(self) -> float:
        return self._time

    def timestep(self) -> int:
        return self._timestep

    def state(self) -> ObsType:
        return self._state

    def control(self) -> Any:
        return self._control

    def physical_control(self) -> Any:
        return self._physical_control

    def terminal(self) -> bool:
        return False

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)


class PhysicsError(RuntimeError):
    pass


class Task(abc.ABC):
    """Defines a task based on a controlled dynamical system."""

    _reference = None
    _observation = None
    _reward = None

    def __init__(self) -> None:
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def target(self, time: float) -> ObsType:
        raise NotImplementedError

    @abc.abstractmethod
    def action_space(self, physics: Physics) -> Space:
        raise NotImplementedError

    @abc.abstractmethod
    def observation_space(self, physics: Physics) -> Space:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, physics: Physics) -> None:
        raise NotImplementedError

    def after_reset(self, physics: Physics) -> None:
        pass

    def before_step(self, action: ActType, physics: Physics) -> None:
        control = action
        physics.set_control(control)

    @abc.abstractmethod
    def step(self, physics: Physics) -> None:
        raise NotImplementedError

    def after_step(self, physics: Physics) -> None:
        pass

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
    """Environment based on physics and task."""

    _buffer = Buffer()

    def __init__(self, physics: Physics, task: Task) -> None:
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
        self._task.after_reset(self._physics)
        time = self._physics.time()
        timestep = self._physics.timestep()
        control = None
        physical_control = None
        state = self._physics.state()
        action = None
        reference = self._task.reference(self._physics)
        observation = self._task.observation(self._physics)
        reward = None
        self._buffer.push(
            time, timestep, control, physical_control, state,
            action, reference, observation, reward,
        )
        return observation

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        if self._buffer.empty():
            raise RuntimeError
        self._task.before_step(action, self._physics)
        self._task.step(self._physics)
        self._task.after_step(self._physics)
        time = self._physics.time()
        timestep = self._physics.timestep()
        control = self._physics.control()
        physical_control = self._physics.physical_control()
        state = self._physics.state()
        reference = self._task.reference(self._physics)
        observation = self._task.observation(self._physics)
        reward = self._task.reward(self._physics)
        done = self._task.done(self._physics)
        info = self._task.info(self._physics)
        self._buffer.push(
            time, timestep, control, physical_control, state,
            action, reference, observation, reward,
        )
        return observation, reward, done, info

    def close(self) -> None:
        self._buffer = Buffer()

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)
        self._physics.seed(seed)
        self._task.seed(seed)
