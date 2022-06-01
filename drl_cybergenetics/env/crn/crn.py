#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typing

import numpy as np
import matplotlib.pyplot as plt

from .env import Env
from .spaces import Discrete, Box
from .wrapper import Wrapper
from .dyn_simulator import DynSimulator, EcoliDynSimulator
from .ref_trajectory import RefTrajectory, ConstantRefTrajectory, BandPassFilter


def make(cls: str, **kwargs):
    if cls == 'CRN-v0':
        return CRN(**kwargs)
    elif cls == 'CRNContinuous-v0':
        return CRNContinuous(**kwargs)
    else:
        raise RuntimeError


class Cache:

    def __init__(self):
        # time, default: np.float32
        self.t = []
        # states, default: np.float32
        self.trajectory = []
        # noise corrupted target observations, default: np.float32
        self.observations = []
        # actions input, default: float
        self.actions = []
        # noise corrupted actions taken, default: float
        self.actions_taken = []
        # rewards measuring the distance target observation and reference trajectory, default: float
        self.rewards = []
        # steps done
        self.steps_done = 0
        # counts of the occurrence that target observation which tracks reference falls in tolerance
        self.aggregator_in_tolerance = []

    def step(self):
        self.steps_done += 1

    def reset(self):
        self.__init__()


class CRN(Env):

    _cache = Cache()

    def __init__(
        self,
        dyn_simulator: DynSimulator = EcoliDynSimulator(),
        ref_trajectory: RefTrajectory = ConstantRefTrajectory(),
        sampling_rate: float = 10,
        system_noise: float = 1e-3,
        observation_noise: float = 1e-3,
        action_noise: float = 1e-3,
    ) -> None:
        super().__init__()
        # dynamics simulator
        self.dyn_simulator = dyn_simulator
        # reference trajectory generator
        self.ref_trajectory = ref_trajectory
        # sampling rate in minutes
        self.T_s = sampling_rate
        # dynamics system noise
        self.system_noise = system_noise
        # target observation noise
        self.observation_noise = observation_noise
        # action taken noise
        self.action_noise = action_noise
        # observation space
        self.observation_space = Box(
            low=self.dyn_simulator.state_min,
            high=self.dyn_simulator.state_max,
            shape=self.dyn_simulator.state_shape,
            dtype=self.dyn_simulator.state_dtype,
        )
        # discrete action space
        self.action_space = Discrete(n=20)  # number of discrete actions

    def reset(self):
        self._cache.reset()
        # time
        self._cache.t.append(np.zeros((1,), dtype=self.observation_space.dtype))  # intitial t = 0
        # state
        state = self.dyn_simulator.init  # initial state
        self._cache.trajectory.append(state)
        # target observation
        observation = state[[self.dyn_simulator.dim_observed]]  # initial target observation
        self._cache.observations.append(observation)
        return state

    def step(self, action: typing.Union[int, np.ndarray], reward_func: str):
        # reset first
        if not self._cache.trajectory:
            raise RuntimeError
        # time
        t = np.array([self.T_s * self._cache.steps_done + self.T_s], dtype=self.observation_space.dtype)  # t + T_s
        self._cache.t.append(t)
        # action
        if isinstance(self.action_space, Discrete):
            action = self.dyn_simulator.action_min + (action + 1) / self.action_space.n * (self.dyn_simulator.action_max - self.dyn_simulator.action_min)
        else:
            action = float(action[0])
        self._cache.actions.append(action)
        # noise corrupted action taken
        action_taken = action + self._rng.normal(0.0, self.action_noise)
        action_taken = min(max(action_taken, self.dyn_simulator.action_min), self.dyn_simulator.action_max)  # clip
        self._cache.actions_taken.append(action_taken)
        # state
        state = self._cache.trajectory[self._cache.steps_done]  # y(t)
        state = self.dyn_simulator(state, self.T_s, action_taken, self.system_noise)  # y(t + T_s)
        self._cache.trajectory.append(state)
        # noise corrupted target observation
        observation = state[[self.dyn_simulator.dim_observed]]  # obs(t + T_s)
        observation += self._rng.normal(0.0, self.observation_noise)
        observation = np.clip(observation, self.dyn_simulator.state_min, self.dyn_simulator.state_max)  # clip
        self._cache.observations.append(observation)
        # reward
        ref, _ = self.ref_trajectory(t)  # ref(t + T_s)
        reward = self._compute_reward(float(observation[0]), float(ref[0]), reward_func)
        self._cache.rewards.append(reward)
        # done
        done = False
        # info
        info = {
            'count_in_tolerance': self._count_in_tolerance(float(state[[self.dyn_simulator.dim_observed]][0]), float(ref[0])),
        }
        # step
        self._cache.step()
        return state, reward, done, info

    def _compute_reward(self, achieved: float, desired: float, func: str) -> float:
        if np.isnan(desired):
            return 0
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
        elif func == 'gaussian':
            reward = float(
                np.exp(
                -(achieved - desired) ** 2 / self.ref_trajectory.tolerance ** 2
                )
            )
        else:
            raise RuntimeError
        return reward

    def _count_in_tolerance(self, achieved: float, desired: float) -> int:
        return int((abs(achieved - desired) / desired < self.ref_trajectory.tolerance))

    def render(self, mode: str = 'human', cache: typing.Optional[Cache] = None):
        # reset first or load cache first
        if (not self._cache.trajectory) and (cache is None):
            raise RuntimeError
        cache = self._cache if cache is None else cache
        # reference trajectory with tolerance margins as time
        delta = 0.1  # simulation sampling rate
        t_ref = np.arange(0, self.T_s * cache.steps_done + delta, delta)
        ref_trajectory, tolerance_margin = self.ref_trajectory(t_ref)
        # states vs. time
        t = np.arange(0, self.T_s * cache.steps_done + self.T_s, self.T_s)
        trajectory = np.stack(cache.trajectory, axis=1)
        # target observations vs. time
        observations = np.concatenate(cache.observations, axis=0)
        # intensity as time
        t_I = np.concatenate([np.arange(self.T_s * i, self.T_s * (i + 1) + 1) for i in range(cache.steps_done)])
        actions = np.array(cache.actions).repeat(self.T_s + 1)
        I = actions * self.dyn_simulator.percentage_thres / 100 * self.dyn_simulator.intensity_thres
        actions_taken = np.array(cache.actions_taken).repeat(self.T_s + 1)
        I_applied = actions_taken * self.dyn_simulator.percentage_thres / 100 * self.dyn_simulator.intensity_thres
        # rewards vs. time
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
            if np.isnan(ref_trajectory).any():
                axs[0].scatter(t_ref, ref_trajectory, color='grey')
                axs[0].errorbar(t_ref, ref_trajectory, yerr=ref_trajectory * self.ref_trajectory.tolerance, color='grey')
            else:
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
            if np.isnan(ref_trajectory).any():
                axs[0, 0].scatter(t_ref, ref_trajectory, color='grey')
                axs[0, 0].errorbar(t_ref, ref_trajectory, yerr=ref_trajectory * self.ref_trajectory.tolerance, color='grey')
            else:
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
            if np.isnan(ref_trajectory).any():
                axs[0, 1].scatter(t_ref, ref_trajectory, color='grey')
                axs[0, 1].errorbar(t_ref, ref_trajectory, yerr=ref_trajectory * self.ref_trajectory.tolerance, color='grey')
            else:
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
        self._cache = Cache()

    def seed(self, seed: typing.Optional[int] = None) -> None:
        self._rng.seed(seed)
        self.dyn_simulator.seed(seed)
        self.action_space.seed(seed)
        #self.observation_space.seed(seed)


class CRNContinuous(CRN):

    def __init__(
        self,
        dyn_simulator: DynSimulator = EcoliDynSimulator(),
        ref_trajectory: RefTrajectory = ConstantRefTrajectory(),
        sampling_rate: float = 10,
        system_noise: float = 1e-3,
        observation_noise: float = 1e-3,
        action_noise: float = 1e-3,
    ) -> None:
        super().__init__(
            dyn_simulator,
            ref_trajectory,
            sampling_rate,
            system_noise,
            observation_noise,
            action_noise,
        )
        # continuous action space, overwriting discrete action space
        self.action_space = Box(
            low=self.dyn_simulator.action_min,
            high=self.dyn_simulator.action_max,
            shape=(1,),
            dtype=np.float32,  # dtype
        )
