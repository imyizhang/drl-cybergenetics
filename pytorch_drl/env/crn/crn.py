#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from .env import Env
from .utils import ConstantRefTrajectory


def make(cls: str, **kwargs):
    if cls == 'CRN-v0':
        return CRN(**kwargs)
    elif cls == 'CRNContinuous-v0':
        return CRNContinuous(**kwargs)
    else:
        raise RuntimeError


# default nominal parameters
d_r = 0.0956
d_p = 0.0214
k_m = 0.0116
b_r = 0.0965

class CRN(Env):
    """A dynamical fold-change model with discrete action space (u), which describes
    the evolution of the species (s) over their initial conditions in the un-induced
    system:

        ds / dt = A_c @ s + B_c @ a

    where

        s = np.array([[R],
                      [P],
                      [G]])

        a = np.array([[1],
                      [u]])

        A_c = np.array([[-d_r, 0.0, 0.0],
                        [d_p + k_m, -d_p - k_m, 0.0],
                        [0.0, d_p, -d_p]])

        B_c = np.array([[d_r, b_r],
                        [0.0, 0.0],
                        [0.0, 0.0]])

    Note that the default nominal parameters are derived from a maximum-likelihood fit

        d_r = 0.0956
        d_p = 0.0214
        k_m = 0.0116
        b_r = 0.0965

    and assuming that the system is at the un-induced steady-state (u = 0) at t = 0,
    the initial condition of this system is R = P = G = 1.

    Reference:
        [1] https://www.nature.com/articles/ncomms12546.pdf
    """

    def __init__(
        self,
        ref_trajectory: typing.Callable[[np.ndarray], typing.Any] = ConstantRefTrajectory(),
        sampling_rate: float = 10,
        observation_noise: float = 1e-3,
        action_noise: float = 1e-3,
        system_noise: float = 1e-3,
        theta: list = [d_r, d_p, k_m, b_r],
        observation_mode: str = 'partially_observed',
        extra_info: bool = False,
    ) -> None:
        super().__init__()
        # reference trajectory generator
        self.ref_trajectory = ref_trajectory
        # sampling rate in minutes
        self._T_s = sampling_rate
        # observation noise
        self._observation_noise = observation_noise
        # action noise
        self._action_noise = action_noise
        # system noise
        self._system_noise = system_noise
        # parameters for continuous-time fold-change model
        self._theta = theta
        self._d_r, self._d_p, self._k_m, self._b_r = self._theta
        self._A_c = np.array([[-self._d_r, 0.0, 0.0],
                              [self._d_p + self._k_m, -self._d_p - self._k_m, 0.0],
                              [0.0, self._d_p, -self._d_p]])
        self._B_c = np.array([[self._d_r, self._b_r],
                              [0.0, 0.0],
                              [0.0, 0.0]])
        # observation mode, either noise corrupted G or perfect R, P, G would be observed by an agent
        self._observation_mode = observation_mode
        # whether observed state includes extra info about tracking reference
        self._extra_info = extra_info
        # initialize the cache
        self._init()

    def _init(self) -> None:
        # actions input
        self._actions = []
        # noise corrupted actions taken
        self._actions_taken = []
        # perfect R, P, G states
        self._trajectory = []
        # noise corrupted G observations
        self._observations = []
        # rewards measuring the distance between perfect G and reference trajectory
        self._rewards = []
        # previous reward
        self._prev_reward = None
        # steps done
        self._steps_done = 0

    @property
    def state(self) -> typing.Optional[np.ndarray]:
        """Current state, which can be partially or fully observed."""
        if self._extra_info and (self._state is not None):
            return np.concatenate((self._state, self._state_in_tolerance,), axis=None)
        return self._state

    @property
    def _state(self) -> typing.Optional[np.ndarray]:
        if self._trajectory and self._observations:
            # noise corrupted G observed
            if self._observation_mode == 'partially_observed':
                return self._observations[self._steps_done]
            # perfect R, P, G observed
            return self._trajectory[self._steps_done]
        return None

    @property
    def state_dim(self) -> int:
        """Dimensions of states observed."""
        if self._extra_info:
            return self._state_dim + 1
        return self._state_dim

    @property
    def _state_dim(self) -> int:
        # noise corrupted G observed
        if self._observation_mode == 'partially_observed':
            return 1  # G
        # perfect R, P, G observed
        return 3  # R, P, G

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
        if self._extra_info:
            state = self._reset()
            return np.concatenate((state, self._state_in_tolerance,), axis=None)
        return self._reset()

    def _reset(self) -> np.ndarray:
        self._init()
        # state
        state = np.ones((3,))  # R = P = G = 1
        self._trajectory.append(state)
        # observation
        observation = state[[2]]  # G = 1
        self._observations.append(observation)
        # noise corrupted G observed
        if self._observation_mode == 'partially_observed':
            return observation
        # perfect R, P, G observed
        return state

    @property
    def _T(self) -> np.ndarray:
        """Current time."""
        return np.array([self._steps_done * self._T_s])

    @property
    def _ref(self) -> np.ndarray:
        """Current reference."""
        return self.ref_trajectory(self._T)[0]

    @property
    def _G(self) -> typing.Optional[np.ndarray]:
        """Current G."""
        # perfect G
        if self._trajectory:
            return self._trajectory[self._steps_done][[2]]
        # or noise corrupted G ?
        #if self._observations:
        #    return self._observations[self._steps_done]
        return None

    @property
    def _in_tolerance(self) -> typing.Optional[bool]:
        """Whether current G which tracks the reference falls in tolerance."""
        if self._G is None:
            return None
        return (abs(self._G - self._ref) / self._ref < self.ref_trajectory.tolerance)

    @property
    def _count_in_tolerance(self) -> typing.Optional[int]:
        """Count of the occurrence that current G which tracks the reference
        falls in tolerance.
        """
        if self._in_tolerance is None:
            return None
        return int(self._in_tolerance)

    @property
    def _state_in_tolerance(self) -> typing.Optional[np.ndarray]:
        """Whether current G which tracks the reference falls in tolerance.

            above tolerance: +1
            in tolerance: 0
            below tolerance: -1
        """
        if self._G is None:
            return None
        if abs(self._G - self._ref) / self._ref < self.ref_trajectory.tolerance:
            return np.array([0])
        elif self._G > self._ref:
            return np.array([1])
        else:
            return np.array([-1])

    def step(
        self,
        action: typing.Union[float, np.ndarray],
        reward_func: str,
    ):
        if self._extra_info:
            state, reward, done, info = self._step(action, reward_func)
            return np.concatenate((state, self._state_in_tolerance,), axis=None), reward, done, info
        return self._step(action, reward_func)

    def _step(
        self,
        action: typing.Union[float, np.ndarray],
        reward_func: str,
    ):
        # reset first
        if self.state is None:
            raise RuntimeError

        # action
        if self.discrete:
            action = (action + 1) / self.action_dim  # float
        else:
            action = action[0]  # float
        self._actions.append(action)

        # noise corrupted action taken
        action += self._rng.normal(0.0, self._action_noise)
        action = np.clip(action, 0.0, 1.0)  # clip to [0.0, 1.0)
        self._actions_taken.append(action)

        # state
        delta = 0.1  # system dynamics simulation sampling rate
        sol = solve_ivp(
            self._func,
            (0, self._T_s + delta),
            self._trajectory[self._steps_done],
            t_eval=np.arange(0, self._T_s + delta, delta),
            args=(action,),
        )  # system dynamics simulation
        state = sol.y[:, -1]
        state += self._rng.normal(0.0, self._system_noise)  # system noise simulation
        state = np.clip(state, 0.0, np.inf)  # clip to [0, inf)
        self._trajectory.append(state)

        # noise corrupted G
        observation = state[[2]]
        observation += self._rng.normal(0.0, self._observation_noise)
        observation = np.clip(observation, 0.0, np.inf)  # clip to [0, inf)
        self._observations.append(observation)

        # step
        self._steps_done += 1

        # reward
        reward = self._compute_reward(self._G[0], self._ref[0], reward_func)
        self._rewards.append(reward)

        # done
        done = False

        # info
        info = {
            'action_taken': action,
            'state': state,
            'observation': observation,
            'count_in_tolerance': self._count_in_tolerance,
        }

        # what if returned state containing tracking info?
        # noise corrupted G observed
        if self._observation_mode == 'partially_observed':
            return observation, reward, done, info
        # perfect R, P, G observed
        return state, reward, done, info

    def _func(self, t: float, y: np.ndarray, action: float) -> np.ndarray:
        a = np.array([1.0, action])
        return self._A_c @ y + self._B_c @ a

    def _compute_reward(
        self,
        achieved_goal: float,
        desired_goal: float,
        func: str,
    ) -> float:
        absolute_error = abs(desired_goal - achieved_goal)
        relative_error = absolute_error / desired_goal
        squared_error = absolute_error ** 2
        abs_logarithmic_error = abs(np.log(desired_goal + 1) - np.log(achieved_goal + 1))
        squared_logarithmic_error = (abs_logarithmic_error) ** 2
        reward = 0.0
        if func == 'negative_se':
            reward -= squared_error
        elif func == 'negative_sle':
            reward -= squared_logarithmic_error
        elif func == 'negative_ale':
            reward -= absolute_logarithmic_error
        elif func == 'negative_ae':
            reward -= absolute_error
        elif func == 'negative_logae':
            reward -= np.log(absolute_error)
        elif func == 'negative_expae':
            reward -= np.exp(absolute_error)
        elif func == 'inverse_ae':
            reward = 1.0 / absolute_error
        elif func == 'negative_re':
            reward -= relative_error
        elif func == 'negative_sqrtre':
            reward -= relative_error ** 0.5
        elif func == 'in_tolerance':
            reward = 1.0 if self._in_tolerance else 0.0
        elif func == 'scaled_se':
            reward = -squared_error * 100 + self._count_in_tolerance * 10
        elif func == 'scaled_sle':
            reward = -squared_logarithmic_error * 100 + self._count_in_tolerance * 10
        elif func == 'complex':
            error = -squared_error * 100 + self._count_in_tolerance * 10
            if self._prev_reward is not None:
                reward = error - self._prev_reward
            self._prev_reward = error
            reward -= relative_error ** 0.5
        else:
            raise RuntimeError
        return reward

    def render(
        self,
        render_mode: str = 'human',
        actions: typing.Optional[typing.List] = None,
        actions_taken: typing.Optional[typing.List] = None,
        trajectory: typing.Optional[typing.List] = None,
        observations: typing.Optional[typing.List] = None,
        rewards: typing.Optional[typing.List] = None,
        steps_done: typing.Optional[int] = None
    ) -> None:
        # check if replay
        replay = not ((not actions) \
                      and (not actions_taken) \
                      and (not trajectory) \
                      and (not observations) \
                      and (not rewards) \
                      or (not steps_done))
        # reset first
        if (self.state is None) and (not replay):
            raise RuntimeError

        # actions input
        _actions = actions if replay else self._actions
        # noise corrupted actions taken
        _actions_taken = actions_taken if replay else self._actions_taken
        # perfect R, P, G states
        _trajectory = trajectory if replay else self._trajectory
        # noise corrupted G observations
        _observations = observations if replay else self._observations
        # rewards measuring the distance between perfect G and reference trajectory
        _rewards = rewards if replay else self._rewards
        # steps done
        _steps_done = steps_done if replay else self._steps_done

        # reference trajectory and tolerance margin
        delta = 0.1  # simulation sampling rate
        t = np.arange(0, self._T_s * _steps_done + delta, delta)
        ref_trajectory, tolerance_margin = self.ref_trajectory(t)
        # species
        T = np.arange(0, self._T_s * _steps_done + self._T_s, self._T_s)
        R, P, G = np.stack(_trajectory, axis=1)
        # fluorescent observed
        G_observed = np.concatenate(_observations, axis=0)
        # intensity
        t_u = np.concatenate([
            np.arange(self._T_s * i, self._T_s * (i + 1) + 1) for i in range(_steps_done)
        ])
        u = np.array(_actions).repeat(self._T_s + 1) * 100
        u_applied = np.array(_actions_taken).repeat(self._T_s + 1) * 100
        # reward
        reward = np.array(_rewards)

        # plot colors
        c_R, c_P, c_G = ['tab:red', 'purple', 'green']
        c_u = 'tab:blue'
        c_reward = 'tab:orange'
        # experimental observation, partially shown
        if render_mode == 'human':
            fig, axs = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(10, 5),
                sharex=True,
                gridspec_kw={'height_ratios': [2, 1]}
            )
            fig.tight_layout()
            # subplot fluorescent
            axs[0].plot(t, ref_trajectory, '--', color='grey')
            axs[0].fill_between(t, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            axs[0].plot(T, G_observed, 'o--', label='G observed', color=c_G, alpha=0.5)
            axs[0].set_ylabel('concentration fold change (1/min)')
            axs[0].legend(framealpha=0.2)
            # subplot intensity
            axs[1].plot(t_u, u, '-', label='u', color=c_u)
            axs[1].set_xlabel('Time (min)')
            axs[1].set_ylabel('intensity (%)')
            axs[1].legend(framealpha=0.2)
            plt.show()
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
            # subplot species
            axs[0, 0].plot(t, ref_trajectory, '--', color='grey')
            axs[0, 0].fill_between(t, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            axs[0, 0].plot(T, R, 'o-', label='R', color=c_R)
            axs[0, 0].plot(T, P, 'o-', label='P', color=c_P)
            axs[0, 0].plot(T, G, 'o-', label='G', color=c_G)
            axs[0, 0].set_ylabel('concentration fold change (1/min)')
            axs[0, 0].legend(framealpha=0.2)
            # subplot intensity
            axs[1, 0].plot(t_u, u, '-', label='u', color=c_u)
            axs[1, 0].plot(t_u, u_applied, '--', label='u applied', color=c_u, alpha=0.5)
            axs[1, 0].set_xlabel('Time (min)')
            axs[1, 0].set_ylabel('intensity (%)')
            axs[1, 0].legend(framealpha=0.2)
            # subplot fluorescent
            axs[0, 1].plot(t, ref_trajectory, '--', color='grey')
            axs[0, 1].fill_between(t, tolerance_margin[0], tolerance_margin[1], color='grey', alpha=0.2)
            axs[0, 1].plot(T, G, 'o-', label='G', color=c_G)
            axs[0, 1].plot(T, G_observed, 'o--', label='G observed', color=c_G, alpha=0.5)
            axs[0, 1].set_ylabel('concentration fold change (1/min)')
            axs[0, 1].legend(framealpha=0.2)
            # subplot reward
            axs[1, 1].plot(T[1:], reward, color=c_reward)
            axs[1, 1].set_xlabel('Time (min)')
            axs[1, 1].set_ylabel('reward')
            plt.show()

    def close(self) -> None:
        self._init()


class CRNContinuous(CRN):
    """A dynamical fold-change model with continuous action space (u)."""

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
