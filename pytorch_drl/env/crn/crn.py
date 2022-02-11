#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np
import matplotlib.pyplot as plt

from .env import Env
from .dyn_simulator import DynSimulator, EcoliDynSimulator
from .ref_trajectory import RefTrajectory, ConstantRefTrajectory


def make(cls: str, **kwargs):
    if cls == 'CRN-v0':
        return CRN(**kwargs)
    elif cls == 'CRNContinuous-v0':
        return CRNContinuous(**kwargs)
    else:
        raise RuntimeError


class Cache(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'Cache' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value


class CRN(Env):

    _cache = Cache()

    def __init__(
        self,
        ref_trajectory: RefTrajectory = ConstantRefTrajectory(),
        dyn_simulator: DynSimulator = EcoliDynSimulator(),
        sampling_rate: float = 10,
        system_noise: float = 1e-3,
        observation_noise: float = 1e-3,
        action_noise: float = 1e-3,
        observation_mode: str = 'partially_observed',
        extra_info: bool = False,
    ) -> None:
        super().__init__()
        # reference trajectory generator
        self.ref_trajectory = ref_trajectory
        # dynamics simulator
        self.dyn_simulator = dyn_simulator
        # sampling rate in minutes
        self.T_s = sampling_rate
        # dynamics system noise
        self.system_noise = system_noise
        # target observation noise
        self.observation_noise = observation_noise
        # action taken noise
        self.action_noise = action_noise
        # observation mode, either noise corrupted target observation or perfect state would be observed
        self.observation_mode = observation_mode
        # whether observation includes extra info about tracking reference
        self.extra_info = extra_info
        # initialize the cache
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        # states
        self._cache.trajectory = []
        # noise corrupted target observations
        self._cache.observations = []
        # actions input
        self._cache.actions = []
        # noise corrupted actions taken
        self._cache.actions_taken = []
        # rewards measuring the distance between target observation and reference trajectory
        self._cache.rewards = []
        # steps done
        self._cache.steps_done = 0
        # counts of the occurrence that target observation which tracks reference falls in tolerance
        self._cache.aggregator_in_tolerance = []

    def seed(self, seed: typing.Optional[int] = None):
        self._rng.seed(seed)
        self.dyn_simulator._rng.seed(seed)

    @property
    def state(self) -> typing.Optional[np.ndarray]:
        """Current state observed, which can be partially or fully observed."""
        # TODO: extra info
        if self.extra_info and (self._state is not None):
            return np.concatenate((self._state, self.state_in_tolerance,), axis=None)
        return self._state

    @property
    def _state(self) -> typing.Optional[np.ndarray]:
        if self._cache.trajectory and self._cache.observations:
            # current noise corrupted target observation
            if self.observation_mode == 'partially_observed':
                return self._cache.observations[self._cache.steps_done]
            # current perfect state
            return self._cache.trajectory[self._cache.steps_done]
        return None

    @property
    def state_dim(self) -> int:
        """Dimensions of state observed."""
        # TODO: extra info
        if self.extra_info:
            return self._state_dim + 1
        return self._state_dim

    @property
    def _state_dim(self) -> int:
        # dimension of noise corrupted target observation
        if self.observation_mode == 'partially_observed':
            return 1
        # dimension of perfect state
        return self.dyn_simulator.state_dim

    @property
    def state_in_tolerance(self) -> typing.Optional[np.ndarray]:
        """Whether target observation (achieved) that tracks reference (desired)
        falls in tolerance:

            above tolerance: +1
            in tolerance: 0
            below tolerance: -1
        """
        if self._state is None:
            return None
        achieved = self._cache.observations[self._cache.steps_done][0]
        desired = self.ref_trajectory(np.array([self.T_s * self._cache.steps_done]))[0][0]
        if abs(achieved - desired) / desired < self.ref_trajectory.tolerance:
            return np.array([0])
        elif achieved > desired :
            return np.array([1])
        else:
            return np.array([-1])

    @property
    def discrete(self) -> bool:
        """Whether action space is discrete."""
        return True

    @property
    def action_dim(self) -> int:
        """Dimensions of action space."""
        # discrete action space
        return 20

    def action_sample(self) -> int:
        """Sample from action space."""
        # discrete action space
        return self._rng.randint(0, self.action_dim)

    def reset(self) -> np.ndarray:
        self._initialize_cache()
        # state
        state = self.dyn_simulator.init  # initial state
        self._cache.trajectory.append(state)
        # observation
        observation = state[[self.dyn_simulator.dim_observed]]  # initial target observation
        self._cache.observations.append(observation)
        return self.state

    def step(self, action: typing.Union[int, np.ndarray], reward_func: str):
        # reset first
        if self.state is None:
            raise RuntimeError
        # action
        action = (action + 1) / self.action_dim if self.discrete else action[0]  # float
        self._cache.actions.append(action)
        # noise corrupted action taken
        action_taken = action + self._rng.normal(0.0, self.action_noise)
        action_taken = np.clip(action_taken, 0.0, 1.0)  # clip to [0.0, 1.0)
        self._cache.actions_taken.append(action_taken)
        # state
        state = self._cache.trajectory[self._cache.steps_done]  # y(t)
        state = self.dyn_simulator(state, self.T_s, action_taken, self.system_noise)  # y(t + T_s)
        self._cache.trajectory.append(state)
        # noise corrupted target observation
        observation = state[[self.dyn_simulator.dim_observed]]
        observation += self._rng.normal(0.0, self.observation_noise)
        observation = np.clip(observation, 0.0, np.inf)  # clip to [0, inf)
        self._cache.observations.append(observation)
        # previously achieved
        #self._prev_achieved = self._cache.observations[self._cache.steps_done]
        # step
        self._cache.steps_done += 1
        # reward
        ref = self.ref_trajectory(np.array([self.T_s * self._cache.steps_done]))[0]
        reward = self._compute_reward(observation[0], ref[0], reward_func)
        self._cache.rewards.append(reward)
        # done
        done = False
        # info
        info = {
            'action': action,
            'action_taken': action_taken,
            'state': state,
            'observation': observation,
            'count_in_tolerance': self._count_in_tolerance(state[[self.dyn_simulator.dim_observed]][0], ref[0]),
        }
        return self.state, reward, done, info

    def _compute_reward(self, achieved: float, desired: float, func: str) -> float:
        p = 1
        if func == 'neg_error':
            reward = -abs(achieved - desired) ** p
        elif func == 'inverse_error':
            reward = abs(achieved - desired) ** (-p)
        elif func == 'neg_relative_error':
            reward = -abs(achieved - desired) / desired
        elif func == 'in_tolerance':
            reward = self._count_in_tolerance(achieved, desired)
        elif func == 'percentage_in_tolerance':
            self._cache.aggregator_in_tolerance.append(self._count_in_tolerance(achieved, desired))
            reward = float(np.array(self._cache.aggregator_in_tolerance).mean())
        elif func == 'scaled_error':
            reward = -abs(achieved - desired) ** p * 100 + self._count_in_tolerance(achieved, desired) * 10
        else:
            raise RuntimeError
        return reward

    def _count_in_tolerance(self, achieved: float, desired: float) -> int:
        return int((abs(achieved - desired) / desired < self.ref_trajectory.tolerance))

    def render(self, mode: str = 'human', cache: typing.Optional[Cache] = None) -> None:
        # reset first or load cache first
        if (self.state is None) and (not cache):
            raise RuntimeError
        # cache
        if cache:
            cache.trajectory = [self.dyn_simulator.init,] + cache.trajectory
            cache.observations = [self.dyn_simulator.init[[self.dyn_simulator.dim_observed]],] + cache.observations
        else:
            cache = self._cache
        # subplot: reference trajectory with tolerance margins as time
        delta = 0.1  # simulation sampling rate
        t_ref = np.arange(0, self.T_s * cache.steps_done + delta, delta)
        ref_trajectory, tolerance_margin = self.ref_trajectory(t_ref)
        # & states vs. time
        t = np.arange(0, self.T_s * cache.steps_done + self.T_s, self.T_s)
        trajectory = np.stack(cache.trajectory, axis=1)
        # subplot: target observations vs. time
        observations = np.concatenate(cache.observations, axis=0)
        # subplot: intensity as time
        t_I = np.concatenate([np.arange(self.T_s * i, self.T_s * (i + 1) + 1) for i in range(cache.steps_done)])
        actions = np.array(cache.actions).repeat(self.T_s + 1)
        I = actions * self.dyn_simulator.percentage_thres / 100 * self.dyn_simulator.intensity_thres
        actions_taken = np.array(cache.actions_taken).repeat(self.T_s + 1)
        I_applied = actions_taken * self.dyn_simulator.percentage_thres / 100 * self.dyn_simulator.intensity_thres
        # subplot: rewards vs. time
        rewards = np.array(cache.rewards)
        # plot settings
        colors = self.dyn_simulator.state_colors
        labels = self.dyn_simulator.state_labels
        color_I = 'tab:blue'
        color_r = 'tab:orange'
        # experimental observation, partially shown
        if mode == 'human':
            fig, axs = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(10, 5),
                sharex=True,
                gridspec_kw={'height_ratios': [2, 1]}
            )
            fig.tight_layout()
            # subplot: reference trajectory with tolerance margins as time
            axs[0].plot(t_ref, ref_trajectory, '--', color='grey')
            axs[0].fill_between(t_ref, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            # & target observations vs. time
            axs[0].plot(t, observations, 'o--', label=labels[self.dyn_simulator.dim_observed], color=colors[self.dyn_simulator.dim_observed], alpha=0.5)
            axs[0].legend(framealpha=0.2)
            axs[0].set_ylabel('')
            # subplot: intensity as time
            axs[1].plot(t_I, I, '-', label='I', color=color_I)
            axs[1].legend(framealpha=0.2)
            axs[1].set_ylabel('intensity')
            axs[1].set_xlabel('Time (min)')
        # dashboard, fully shown
        else:
            fig, axs = plt.subplots(
                nrows=2,
                ncols=2,
                figsize=(10, 5),
                sharex=True,
                gridspec_kw={'height_ratios': [2, 1]}
            )
            fig.tight_layout()
            # subplot: reference trajectory with tolerance margins as time
            axs[0, 0].plot(t_ref, ref_trajectory, '--', color='grey')
            axs[0, 0].fill_between(t_ref, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            # & states vs. time
            for i, (label, color) in enumerate(zip(labels, colors)):
                axs[0, 0].plot(t, trajectory[i], 'o-', label=label, color=color)
            axs[0, 0].legend(framealpha=0.2)
            axs[0, 0].set_ylabel('')
            # subplot: intensity as time
            axs[1, 0].plot(t_I, I, '-', label='I', color=color_I)
            axs[1, 0].plot(t_I, I_applied, '--', label='I applied', color=color_I, alpha=0.5)
            axs[1, 0].legend(framealpha=0.2)
            axs[1, 0].set_ylabel('intensity')
            axs[1, 0].set_xlabel('Time (min)')
            # subplot: reference trajectory with tolerance margins as time
            axs[0, 1].plot(t_ref, ref_trajectory, '--', color='grey')
            axs[0, 1].fill_between(t_ref, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            # & target observations vs. time
            axs[0, 1].plot(t, trajectory[self.dyn_simulator.dim_observed], 'o-', label=labels[self.dyn_simulator.dim_observed], color=colors[self.dyn_simulator.dim_observed])
            axs[0, 1].plot(t, observations, 'o--', label=labels[self.dyn_simulator.dim_observed] + ' observed', color=colors[self.dyn_simulator.dim_observed], alpha=0.5)
            axs[0, 1].legend(framealpha=0.2)
            axs[0, 1].set_ylabel('')
            # subplot: rewards vs. time
            axs[1, 1].plot(t[1:], rewards, color=color_r)
            axs[1, 1].set_ylabel('reward')
            axs[1, 1].set_xlabel('Time (min)')
        return fig

    def close(self) -> None:
        #self._cache = Cache()
        self._initialize_cache()


class CRNContinuous(CRN):

    @property
    def discrete(self) -> bool:
        """Whether action space is discrete."""
        return False

    @property
    def action_dim(self) -> int:
        """Dimensions of action space."""
        # continuous action space
        return 1

    def action_sample(self) -> int:
        """Sample from action space."""
        # continuous action space
        return self._rng.uniform(0, 1, (self.action_dim,))
