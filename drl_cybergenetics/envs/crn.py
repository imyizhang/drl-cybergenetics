#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Callable,
    Optional,
    Union,
)

import numpy as np
import scipy as sp
import scipy.signal as signal
import matplotlib.pyplot as plt

from .control import Buffer, Physics, Task, Environment
from .control.spaces import Space, Discrete, Box


# Reference registry for tracking task

def _const(t, scale):
    "Constant wave."
    return scale + np.zeros_like(t)


def _square(t, scale, amplitude, period, phase):
    "Square wave."
    return scale + amplitude * signal.square(2 * np.pi * t / period + phase).astype(t.dtype)


def _sin(t, scale, amplitude, period, phase):
    "Sine or Cosine wave."
    return scale + amplitude * np.sin(2 * np.pi * t / period + phase).astype(t.dtype)


def _bpf(t, switches):
    "Band-pass filter (BPF)."
    y = np.zeros_like(t)
    mask_nan = True
    for i in range(switches.shape[0]):
        mask = (t == switches[i, 0])
        np.place(y, mask, switches[i, 1])
        mask_nan &= (1 - mask)
    np.place(y, mask_nan, np.nan)
    return y


registered_reference = {
    'const': _const,
    'square': _square,
    'sin': _sin,
    'bpf': _bpf,
}


def register_reference(func: Callable, name: str) -> None:
    "Customize reference to be tracked."
    registered_reference[name] = func


# Reward registry for tracking task

def _inverse_ae(achieved, desired, tolerance, n=1.0):
    "Inverse of absolute error (AE)."
    return float(np.abs(achieved - desired) ** (-n))


def _negative_ae(achieved, desired, tolerance, n=1.0):
    "Negative absolute error (AE)."
    return float(-np.abs(achieved - desired) ** n)


def _negative_re(achieved, desired, tolerance):
    "Negative relative error (RE)."
    return float(-np.abs(achieved - desired) / desired)


def _in_tolerance(achieved, desired, tolerance):
    "Whether falling within tolerance."
    return float(np.abs(achieved - desired) / desired < tolerance)


def _gauss(achieved, desired, tolerance):
    "Gauss."
    return float(np.exp(-0.5 * (achieved - desired) ** 2 / tolerance ** 2))


def _scaled_combination(achieved, desired, tolerance, a=100.0, b=10.0):
    "Scaled combination of errors."
    return _negative_ae(achieved, desired, tolerance) * a \
        + _in_tolerance(achieved, desired, tolerance) * b


registered_reward = {
    'inverse_ae': _inverse_ae,
    'negative_ae': _negative_ae,
    'negative_re': _negative_re,
    'in_tolerance': _in_tolerance,
    'gauss': _gauss,
    'scaled_combination': _scaled_combination,
}


def register_reward(func: Callable, name: str) -> None:
    registered_reward[name] = func


def make(cls: str, **kwargs):
    pass


class CRN(Physics):

    def __init__(
        self,
        ode: Callable,
        ode_kwargs: dict,
        init_state: np.ndarray,
        intensity_thres: float,
        percent_thres: float,
        action_noise: float,
        system_noise: float,
        state_min: Union[np.ndarray, float],
        state_max: Union[np.ndarray, float],
        state_info: dict,
        action_min: Union[np.ndarray, float],
        action_max: Union[np.ndarray, float],
        action_info: dict,
    ) -> None:
        super().__init__()
        self.ode = ode
        self.ode_kwargs = ode_kwargs
        self.init_state = init_state
        self.intensity_thres = intensity_thres
        self.percent_thres = percent_thres
        self.action_noise = action_noise
        self.system_noise = system_noise
        self._state = self.init_state

    def reset(self):
        self._time = 0.0
        self._state = self.init_state
        self._control = None
        self._physical_action = None

    def dynamics(self, time: float, state: np.ndarray, physical_action: float) -> np.ndarray:
        return self.ode(state, physical_action, **self.ode_kwargs) + self._rng.normal(0.0, self.system_noise)

    def set_control(self, action: float) -> None:
        self._control = action * self.percent_thres * self.intensity_thres
        action += self._rng.normal(0.0, self.action_noise)
        action = min(max(action, self.action_min), self.action_max)
        self._physical_action = action * self.percent_thres * self.intensity_thres

    def step(self, sampling_rate: float) -> None:
        # Dynamics simulation sampling rate
        delta = 0.1
        # Dynamics simulation
        sol = sp.integrate.solve_ivp(
            self.dynamics,
            (0, sampling_rate + delta),
            self._state,
            t_eval=np.arange(0, sampling_rate + delta, delta),
            args=(self._physical_action,),
        )
        # ODEs integration solution
        self._state = sol.y[:, -1]
        self._state = np.clip(self._state, self.state_min, self.state_max)
        # Step
        self._time += sampling_rate

    def time(self) -> float:
        return np.array([self._time])

    def state(self) -> np.ndarray:
        return self._state

    def control(self):
        return self._control.astype(self.state_dtype)

    def physical_action(self):
        return self._physical_action.astype(self.state_dtype)


class Track(Task):

    def __init__(
        self,
        reference: Union[Callable, str],
        reference_kwargs: dict,
        reward: Union[Callable, str],
        reward_kwargs: dict,
        sampling_rate: float,
        tolerance: float,
        dim_observed: int,
        observation_noise: float,
    ) -> None:
        super().__init__()
        self.reference_func = reference if callable(reference) else registered_reference[reference]
        self.reference_kwargs = reference_kwargs
        self.reward_func = reward if callable(reward) else registered_reward[reward]
        self.reward_kwargs = reward_kwargs
        self.sampling_rate = sampling_rate
        self.tolerance = tolerance
        self.dim_observed = dim_observed
        self.observation_noise = observation_noise

    def action_space(self, physics: Physics) -> Space:
        return Box(
            low=physics.state_min,
            high=physics.state_max,
            shape=physics.state_shape,
            dtype=physics.state_dtype,
        )

    def observation_space(self, physics: Physics) -> Space:
        return Box(
            low=physics.state_min,
            high=physics.state_max,
            shape=physics.state_shape,
            dtype=physics.state_dtype,
        )

    def before_step(self, action: ActType, physics: Physics) -> None:
        if isinstance(self.action_space, Discrete):
            action = (action + 1) / self.action_space.n
            action = _rescale(action, (0, 1), (physics.action_min, physics.action_max))
        else:
            action = float(action[0])
            action = _rescale(action, (0, 1), (physics.action_min, physics.action_max))
        physics.set_control(action)

    def step(self, physics):
        physics.step(self.sampling_rate)
        self._timestep += 1

    def timestep(self):
        return self._timestep

    def reference(self, physics):
        self._reference = self.reference_func(
            np.array([physics.time]),
            **self.reference_kwargs
        )
        return self._reference

    def observation(self, physics):
        self._observation = physics.state[[self.dim_observed]]
        self._observation += self._rng.normal(0.0, self.observation_noise)
        self._observation = np.clip(self._observation, physics.state_min, physics.state_max)
        return self._observation

    def reward(self, physics):
        self._reward = self.reward_func(self._observation, self._reference, self.tolerance, **self.reward_kwargs)
        return self._reward


class DiscreteTrack(Track):

    def action_space(self, physics: Physics) -> Space:
        return Discrete(n=20)


class CRNEnv(Environment):

    def render(self, mode: str = 'human', buffer: Optional[Buffer] = None):
        # reset first or load buffer first
        if self._buffer.empty() and (buffer is None):
            raise RuntimeError
        buffer = self._buffer if buffer is None else buffer
        tolerance = self._task.tolerance
        sampling_rate = self._task.sampling_rate
        dim = self._task.dim_observed
        # Data: reference trajectory & observation / state  vs. time
        time = np.array(buffer.time)
        reference = np.concatenate(buffer.reference, axis=0)
        observation = np.concatenate(buffer.observation, axis=0)
        state = np.stack(buffer.state, axis=1)
        # Data: control signal vs. time
        time_control = np.concatenate([np.arange(sampling_rate * i, sampling_rate * (i + 2), sampling_rate) for i in range(len(buffer) - 1)])
        control = np.array(buffer.control[1:]).repeat(2)
        physical_action = np.array(buffer.physical_action[1:]).repeat(2)
        # action = np.array(buffer.action[1:]).repeat(2)
        # Data: reward vs. time
        time_reward = time[1:]
        reward = np.array(buffer.reward[1:])
        try:
            import seaborn as sns
            sns.set_theme(style='darkgrid')
        except ImportError:
            pass
        # Partially shown
        if mode == 'human':
            fig, axes = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(10, 5),
                sharex=True,
                gridspec_kw={'height_ratios': [2, 1]},
            )
            fig.tight_layout()
            # Subplot: reference trajectory & observation vs. time
            self.plot_reference(axes[0], time, reference, tolerance)
            self.plot_observation(axes[0], time, observation)
            # Subplot: control signal vs. time
            self.plot_control(axes[1], time_control, control)
            axes[1].set_xlabel('Time (min)')
        # Fully shown
        else:
            fig, axes = plt.subplots(
                nrows=2,
                ncols=2,
                figsize=(10, 5),
                sharex=True,
                gridspec_kw={'height_ratios': [2, 1]}
            )
            fig.tight_layout()
            # Subplot: reference trajectory & state vs. time
            self.plot_reference(axes[0, 0], time, reference, tolerance)
            self.plot_state(axes[0, 0], time, state)
            # Subplot: control signal vs. time
            self.plot_control(axes[1, 0], time_control, control, physical_action)
            axes[1, 0].set_xlabel('Time (min)')
            # Subplot: reference trajectory & observation vs. time
            self.plot_reference(axes[0, 1], time, reference, tolerance)
            self.plot_observation(axes[0, 1], time, observation, state)
            # Subplot: reward vs. time
            self.plot_reward(axes[1, 1], time_reward, reward)
            axes[1, 1].set_xlabel('Time (min)')
        return fig

    @staticmethod
    def plot_state(ax, time, state, **state_info):
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(time, state[i], 'o-', label=label, color=color)
            ax.plot(time[-1], state[-1], marker='.')
        ax.legend(framealpha=0.2)
        ax.set_ylabel('')

    @staticmethod
    def plot_control(ax, time, control, label, color, physical_action=None):
        ax.plot(time, control, '-', label=label, color=color)
        ax.plot(time[-1], control[-1], marker='.', color=color)
        if physical_action is not None:
            ax.plot(time, physical_action, '--', label='I applied', color=color, alpha=0.5)
            ax.plot(time[-1], physical_action[-1], marker='.', color=color)
        ax.legend(framealpha=0.2)
        ax.set_ylabel('intensity')

    @staticmethod
    def plot_reference(ax, time, reference, tolerance=None, color='grey'):
        if np.isnan(reference).any():
            ax.scatter(time, reference, color=color)
            ax.errorbar(time, reference, yerr=reference * tolerance, color=color)
        else:
            ax.plot(time, reference, '--', color=color)
            lower_bound = reference * (1 - tolerance)
            upper_bound = reference * (1 + tolerance)
            ax.fill_between(time, lower_bound, upper_bound, color=color, alpha=0.2)
        ax.set_ylabel('')

    @staticmethod
    def plot_observation(ax, time, observation, label, color):
        ax.plot(time, observation, '-', label=label + ' observed', color=color)
        ax.plot(time, observation, '--', label=label, color=color, alpha=0.5)
        ax.legend(framealpha=0.2)
        ax.set_ylabel('')

    @staticmethod
    def plot_reward(ax, time, reward, color):
        ax.plot(time, reward, color=color)
        ax.set_ylabel('Reward')
