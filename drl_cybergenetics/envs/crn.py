#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union, Type
import pathlib

import numpy as np
import scipy as sp
import scipy.signal as signal
import matplotlib.pyplot as plt
import torch

from .control import Buffer, Physics, Task, Environment
from .control.spaces import Discrete, Box
from .control.wrappers import (
    Wrapper,
    AsTensor,
    LimitTimestep,
    TrackEpisode,
    RecordEpisode,
    FullObservation,
    TimeAwareObservation,
    TimestepAwareObservation,
    ReferenceAwareObservation,
    ToleranceAwareObservation,
    ActionAwareObservation,
    RescaleAction,
)


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


def make(cls: str, kwargs: dict):
    physics = CRN(**kwargs['physics'])
    if cls == 'CRN-v0':
        task = DiscreteTrack(**kwargs['task'])
    elif cls == 'CRNContinuous-v0':
        task = Track(**kwargs['task'])
    else:
        raise RuntimeError
    env = CRNEnv(physics, task, **kwargs['environment'])
    return CRNWrapper(env, **kwargs['wrappers'])


class CRN(Physics):

    def __init__(
        self,
        init_state: np.ndarray,
        system_noise: float = 0.0,
        actuation_noise: float = 0.0,
        state_min: Union[float, np.ndarray] = 0.0,
        state_max: Union[float, np.ndarray] = np.inf,
        state_dtype: Type = np.float32,
        state_info: dict = {},
        control_min: Union[float, np.ndarray] = 0.0,
        control_max: Union[float, np.ndarray] = 1.0,
        control_dtype: Type = float,
        control_info: dict = {},
        ode: Optional[Callable] = None,
        **ode_kwargs
    ) -> None:
        super().__init__()
        self.ode = ode
        self.ode_kwargs = ode_kwargs
        self.init_state = init_state
        self.system_noise = system_noise
        self.actuation_noise = actuation_noise
        self.state_shape = init_state.shape
        self.state_min = state_min.astype(state_dtype) if isinstance(state_min, np.ndarray) else np.full(self.state_shape, state_min, dtype=state_dtype)
        self.state_max = state_max.astype(state_dtype) if isinstance(state_max, np.ndarray) else np.full(self.state_shape, state_max, dtype=state_dtype)
        self.state_dtype = state_dtype
        self.state_info = state_info
        self.control_min = float(control_min)
        self.control_max = float(control_max)
        self.control_dtype = control_dtype
        self.control_info = control_info
        self._state = init_state

    def dynamics(self, time: float, state: np.ndarray, control: float):
        if self.ode is None:
            raise NotImplementedError
        return self.ode(state, control, **self.ode_kwargs) + self._rng.normal(0.0, self.system_noise)

    def reset(self) -> None:
        self._time = 0.0
        self._timestep = 0
        self._state = self.init_state
        self._control = None
        self._physical_control = None

    def set_control(self, control: float) -> None:
        self._control = control
        self._physical_control = self._control + self._rng.normal(0.0, self.actuation_noise)
        self._physical_control = min(max(control, self.control_min), self.control_max)

    def step(self, sampling_rate: float) -> None:
        # Dynamics simulation sampling rate
        delta = 0.1
        # Dynamics simulation
        sol = sp.integrate.solve_ivp(
            self.dynamics,
            (0, sampling_rate + delta),
            self._state,
            t_eval=np.arange(0, sampling_rate + delta, delta),
            args=(self._physical_control,),
        )
        # ODE integration solution
        self._state = sol.y[:, -1]
        self._state = np.clip(self._state, self.state_min, self.state_max)
        # Step
        self._time += sampling_rate
        self._timestep += 1

    def state(self) -> np.ndarray:
        return self._state.astype(self.state_dtype)


class Track(Task):

    def __init__(
        self,
        sampling_rate: float,
        dim_observed: int,
        tolerance: float,
        reward: Union[str, Callable],
        reward_kwargs: dict = {},
        reward_info: dict = {},
        observation_noise: float = 0.0,
        action_min: Union[float, np.ndarray] = -1.0,
        action_max: Union[float, np.ndarray] = 1.0,
        action_dtype: Type = np.float32,
        action_info: dict = {},
        tracking: Optional[Union[Callable, str]] = None,
        **tracking_kwargs
    ) -> None:
        super().__init__()
        self.tracking = tracking if callable(tracking) or tracking is None else registered_reference[tracking]
        self.tracking_kwargs = tracking_kwargs
        self.sampling_rate = sampling_rate
        self.dim_observed = dim_observed
        self.tolerance = tolerance
        self.reward_func = reward if callable(reward) else registered_reward[reward]
        self.reward_kwargs = reward_kwargs
        self.reward_info = reward_info
        self.observation_noise = observation_noise
        self.action_shape = (1,)
        self.action_min = action_min.astype(action_dtype) if isinstance(action_min, np.ndarray) else np.full(self.action_shape, action_min, dtype=action_dtype)
        self.action_max = action_max.astype(action_dtype) if isinstance(action_max, np.ndarray) else np.full(self.action_shape, action_max, dtype=action_dtype)
        self.action_dtype = action_dtype
        self.action_info = action_info

    def target(self, time: np.ndarray):
        if self.tracking is None:
            raise NotImplementedError
        return self.tracking(time, **self.tracking_kwargs)

    def action_space(self, physics: Physics) -> Box:
        return Box(
            low=self.action_min,
            high=self.action_max,
            dtype=self.action_dtype,
        )

    def observation_space(self, physics: Physics) -> Box:
        return Box(
            low=physics.state_min[[self.dim_observed]],
            high=physics.state_max[[self.dim_observed]],
            dtype=physics.state_dtype,
        )

    def reset(self, physics: Physics) -> None:
        self._reference = None
        self._observation = None
        self._reward = None

    def before_step(self, action: Union[int, np.ndarray], physics: Physics) -> None:
        if isinstance(self.action_space(physics), Discrete):
            action = (action + 1) / self.action_space(physics).n
        else:
            action = float(action[0])
            action = (action - float(self.action_min)) / (float(self.action_max) - float(self.action_min))
        control = physics.control_min + action * (physics.control_max - physics.control_min)
        physics.set_control(control)

    def step(self, physics: Physics) -> None:
        physics.step(self.sampling_rate)

    def reference(self, physics: Physics) -> np.ndarray:
        time = np.array([physics.time()]).astype(physics.state_dtype)
        self._reference = self.target(time)
        return self._reference.astype(physics.state_dtype)

    def observation(self, physics: Physics) -> np.ndarray:
        state = physics.state()
        self._observation = state[[self.dim_observed]]
        self._observation += self._rng.normal(0.0, self.observation_noise)
        self._observation = np.clip(self._observation, physics.state_min[[self.dim_observed]], physics.state_max[[self.dim_observed]])
        return self._observation.astype(physics.state_dtype)

    def reward(self, physics: Physics) -> float:
        self._reward = self.reward_func(self._observation, self._reference, self.tolerance, **self.reward_kwargs)
        return self._reward


class DiscreteTrack(Track):

    def action_space(self, physics: Physics) -> Discrete:
        return Discrete(n=20)


class CRNEnv(Environment):

    def render(self, mode: str = 'human', buffer: Optional[Buffer] = None):
        if self._buffer.empty() and (buffer is None):
            raise RuntimeError
        buffer = self._buffer if buffer is None else buffer
        tolerance = self._task.tolerance
        sampling_rate = self._task.sampling_rate
        dim_observed = self._task.dim_observed
        # Data: reference trajectory & state / observation  vs. time
        time = np.array(buffer.time)
        state = np.stack(buffer.state, axis=1)
        observation = np.concatenate(buffer.observation, axis=0)
        delta = 0.1  # simulation sampling rate
        time_reference = np.arange(0, time[-1] + delta, delta)
        reference = self.task.target(time_reference)
        # Data: control signal vs. time
        time_control = np.concatenate([np.arange(sampling_rate * i, sampling_rate * (i + 2), sampling_rate) for i in range(len(buffer) - 1)])
        control = np.array(buffer.control[1:]).repeat(2)
        physical_control = np.array(buffer.physical_control[1:]).repeat(2)
        # Data: reward vs. time
        time_reward = time[1:]
        reward = np.array(buffer.reward[1:])
        # Info: reference trajectory & state / observation  vs. time
        state_info = self._physics.state_info
        observation_info = {
            'color': state_info['color'][dim_observed],
            'label': state_info['label'][dim_observed],
            'ylim': state_info['ylim'],
        }
        # Info: control signal vs. time
        control_info = self._physics.control_info
        # Info: reward vs. time
        reward_info = self._task.reward_info
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
                figsize=(7, 5),
                sharex=True,
                gridspec_kw={'height_ratios': [2, 1]},
            )
            fig.tight_layout()
            # Subplot: reference trajectory & observation vs. time
            self.plot_reference(axes[0], time_reference, reference, tolerance)
            self.plot_observation(axes[0], time, observation, state=None, **observation_info)
            # Subplot: control signal vs. time
            self.plot_control(axes[1], time_control, control, physical_control=None, **control_info)
            axes[1].set_xlabel('Time (min)')
            plt.close()
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
            self.plot_reference(axes[0, 0], time_reference, reference, tolerance)
            self.plot_state(axes[0, 0], time, state, **state_info)
            # Subplot: control signal vs. time
            self.plot_control(axes[1, 0], time_control, control, physical_control, **control_info)
            axes[1, 0].set_xlabel('Time (min)')
            # Subplot: reference trajectory & observation vs. time
            self.plot_reference(axes[0, 1], time_reference, reference, tolerance)
            self.plot_observation(axes[0, 1], time, observation, state[dim_observed], **observation_info)
            # Subplot: reward vs. time
            self.plot_reward(axes[1, 1], time_reward, reward, **reward_info)
            axes[1, 1].set_xlabel('Time (min)')
            plt.close()
        return fig

    @staticmethod
    def plot_state(ax, time, state, color, label, ylim):
        for i in range(state.shape[0]):
            ax.plot(time, state[i], '-', color=color[i], label=label[i], alpha=0.85)
            if len(time) > 0:
                ax.plot(time[-1], state[i][-1], marker='.', color=color[i])
        ax.legend(loc='upper right', framealpha=0.2)
        ax.set_ylim(ylim)
        ax.set_ylabel('')

    @staticmethod
    def plot_control(ax, time, control, physical_control, color, label, ylim):
        ax.plot(time, control, '-', color=color, label=label, alpha=0.85)
        if len(time) > 0:
            ax.plot(time[-1], control[-1], marker='.', color=color)
        if physical_control is not None:
            ax.plot(time, physical_control, '--', color=color, label=label + ' performed', alpha=0.35)
            if len(time) > 0:
                ax.plot(time[-1], physical_control[-1], marker='.', color=color, alpha=0.5)
        ax.legend(loc='upper right', framealpha=0.2)
        ax.set_ylim(ylim)
        ax.set_ylabel('')

    @staticmethod
    def plot_reference(ax, time, reference, tolerance, color='grey'):
        if np.isnan(reference).any():
            ax.plot(time, reference, 'x-', color=color)
            #ax.scatter(time, reference, color=color)
            ax.errorbar(time, reference, yerr=reference * tolerance, color=color)
        else:
            ax.plot(time, reference, '--', color=color)
            ax.fill_between(time, reference * (1 - tolerance), reference * (1 + tolerance), color=color, alpha=0.15)
        ax.set_ylabel('')

    @staticmethod
    def plot_observation(ax, time, observation, state, color, label, ylim):
        ax.plot(time, observation, '-', color=color, label=label + ' observed', alpha=0.85)
        if len(time) > 0:
            ax.plot(time[-1], observation[-1], marker='.', color=color)
        if state is not None:
            ax.plot(time, state, '--', color=color, label=label, alpha=0.35)
            if len(time) > 0:
                ax.plot(time[-1], state[-1], marker='.', color=color, alpha=0.5)
        ax.legend(loc='upper right', framealpha=0.2)
        ax.set_ylim(ylim)
        ax.set_ylabel('')

    @staticmethod
    def plot_reward(ax, time, reward, color, label, ylim):
        ax.plot(time, reward, color=color, label=label + ' reward', alpha=0.85)
        if len(time) > 0:
            ax.plot(time[-1], reward[-1], marker='.', color=color)
        ax.legend(loc='upper right', framealpha=0.2)
        ax.set_ylim(ylim)
        ax.set_ylabel('')


class CRNWrapper(Wrapper):

    def __init__(
        self,
        env: CRNEnv,
        max_episode_steps: Optional[int] = None,
        full_observation: bool = False,
        time_aware: bool = False,
        timestep_aware: bool = False,
        reference_aware: bool = False,
        tolerance_aware: bool = False,
        action_aware: bool = False,
        rescale_action: bool = False,
        action_min: Union[float, np.ndarray] = 0.0,
        action_max: Union[float, np.ndarray] = 1.0,
        track_episode: bool = False,
        record_episode: bool = False,
        path: Union[str, pathlib.Path] = 'episode',
        as_tensor: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        **render_kwargs,
    ):
        if max_episode_steps is not None:
            env = LimitTimestep(env, max_episode_steps)
        if full_observation:
            env = FullObservation(env)
        if time_aware:
            env = TimeAwareObservation(env)
        if timestep_aware:
            env = TimestepAwareObservation(env)
        if reference_aware:
            env = ReferenceAwareObservation(env)
        if tolerance_aware:
            env = ToleranceAwareObservation(env)
        if action_aware:
            env = ActionAwareObservation(env)
        if rescale_action:
            env = RescaleAction(env, action_min, action_max)
        if track_episode:
            env = TrackEpisode(env)
        if record_episode:
            env = RecordEpisode(env, path, **render_kwargs)
        if as_tensor:
            env = AsTensor(env, dtype, device)
        super().__init__(env)
