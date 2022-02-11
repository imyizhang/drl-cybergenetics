#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np
import scipy as sp


class DynSimulator(abc.ABC):
    """Dynamics simulator."""

    def __init__(self, theta: np.ndarray) -> None:
        self.theta = theta
        self._rng = np.random.RandomState(seed=None)

    def seed(self, seed: typing.Optional[int] = None):
        self._rng.seed(seed)

    @property
    def init(self) -> None:
        """Initial y(0)."""
        return None

    @abc.abstractmethod
    def odes(self, t: float, y: np.ndarray) -> np.ndarray:
        """Define ODEs: dy / dt = f(y, t)."""
        raise NotImplementedError

    def __call__(self, y_t: np.ndarray, T_s: float) -> np.ndarray:
        """Given y(t), estimate y(t + T_s) with ODEs."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


# default parameters
d_r = 0.0956
d_p = 0.0214
k_m = 0.0116
b_r = 0.0965

class EcoliDynSimulator(DynSimulator):
    """A dynamical fold-change model with an additional steady-state fold change (u),
    which describes the evolution of the species (s) over their initial conditions
    in the un-induced system:

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
    the initial condition of this system is R = P = G = 1, the additional steady-state
    fold change u = f(U), which a given light intensity, U, can achieve.

    Reference:
        [1] https://www.nature.com/articles/ncomms12546
    """

    def __init__(
        self,
        theta: np.ndarray = np.array([d_r, d_p, k_m, b_r]),
        intensity_thres: float = 1.0,
        percentage_thres: float = 20.0,
    ) -> None:
        super().__init__(theta)
        self._d_r, self._d_p, self._k_m, self._b_r = self.theta
        self._A_c = np.array([
            [-self._d_r, 0.0, 0.0],
            [self._d_p + self._k_m, -self._d_p - self._k_m, 0.0],
            [0.0, self._d_p, -self._d_p]
        ])
        self._B_c = np.array([
            [self._d_r, self._b_r],
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        self.intensity_thres = intensity_thres
        self.percentage_thres = percentage_thres

    @property
    def init(self) -> np.ndarray:
        return np.ones((3,))

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def dim_observed(self) -> int:
        return -1

    @property
    def state_colors(self) -> list:
        return ['tab:red', 'tab:purple', 'tab:green']

    @property
    def state_labels(self) -> list:
        return ['R', 'P', 'G']

    def odes(self, t: float, y: np.ndarray, u: float, eps: float) -> np.ndarray:
        a = np.array([1.0, u])
        # ODEs
        dydt = self._A_c @ y + self._B_c @ a + self._rng.normal(0.0, eps)
        return dydt

    def __call__(self, y_t: np.ndarray, T_s: float, action: float, eps: float) -> np.ndarray:
        U = action * self.percentage_thres * self.intensity_thres  # intensity (%)
        u = self.dose_response(U)  # steady-state fold change, u = f(U)
        delta = 0.1  # dynamics simulation sampling rate
        sol = sp.integrate.solve_ivp(
            self.odes,
            (0, T_s + delta),
            y_t,
            t_eval=np.arange(0, T_s + delta, delta),
            args=(u, eps,),
        )  # dynamics simulation
        state = sol.y[:, -1]  # ODEs solution
        state = np.clip(state, 0.0, np.inf)  # clip to [0, inf)
        return state

    @staticmethod
    def dose_response(U: typing.Union[np.ndarray, float]):
        """Dose-response characterization of the system (the relation between
        constantly applied light intensity (%) and steady-state). Note that
        U = 0 should yield u = 0 at t = 0.
        """
        u = 5.134 / (1 + 5.411 * np.exp(-0.0698 * U)) + 0.1992 - 1
        return u


# default parameters
TF_tot = 2000
k_on = 0.0016399
k_off = 0.34393
k_max = 13.588
K_d = 956.75
n = 4.203
k_basal = 0.02612
k_degR = 0.042116
k_trans = 1.4514
k_degP = 0.007

class YeastDynSimulator(DynSimulator):
    """A ODE-based model describing VP-EL222 mediated gene expression:

        dTF_on / dt = I * k_on * (TF_tot - TF_on) - k_off * TF_on
        dmRNA / dt = k_basal + k_max * (TF_on ^ n) / (K_d ^ n + TF_on ^ n) - k_degR * mRNA
        dProtein / dt = k_trans * mRNA - k_degP * Protein

    where

        TF_tot = 2000
        k_on = 0.0016399
        k_off = 0.34393
        k_max = 13.588
        K_d = 956.75
        n = 4.203
        k_basal = 0.02612
        k_degR = 0.042116
        k_trans = 1.4514
        k_degP = 0.007

    Reference:
        [1] https://www.nature.com/articles/s41467-018-05882-2
    """

    def __init__(
        self,
        theta: np.ndarray = np.array([TF_tot, k_on, k_off, k_max, K_d, n, k_basal, k_degR, k_trans, k_degP]),
        intensity_thres: float = 400.0,
        percentage_thres: float = 20.0,
    ) -> None:
        super().__init__(theta)
        self._TF_tot, \
        self._k_on, \
        self._k_off, \
        self._k_max, \
        self._K_d, \
        self._n, \
        self._k_basal, \
        self._k_degR, \
        self._k_trans, \
        self._k_degP = self.theta
        self.intensity_thres = intensity_thres
        self.percentage_thres = percentage_thres

    @property
    def init(self) -> np.ndarray:
        return np.array([
            0.0,
            self._k_basal / self._k_degR,
            (self._k_trans * self._k_basal) / (self._k_degP * self._k_degR),
        ])

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def dim_observed(self) -> int:
        return -1

    @property
    def state_colors(self) -> list:
        return ['tab:red', 'tab:purple', 'tab:green']

    @property
    def state_labels(self) -> list:
        return ['TF_on', 'mRNA', 'Protein']

    def odes(self, t: float, y: np.ndarray, I: float, eps: float) -> np.ndarray:
        TF_on, mRNA, Protein = y
        # ODEs
        dTF_ondt = I * self._k_on * (self._TF_tot - TF_on) - self._k_off * TF_on
        dmRNAdt = self._k_basal + self._k_max * (TF_on ** self._n) / (self._K_d ** self._n + TF_on ** self._n) - self._k_degR * mRNA
        dProteindt = self._k_trans * mRNA - self._k_degP * Protein
        return np.array([dTF_ondt, dmRNAdt, dProteindt])

    def __call__(self, y_t: np.ndarray, T_s:float, action:float, eps:float) -> np.ndarray:
        I = action * self.percentage_thres / 100 * self.intensity_thres  # intensity
        delta = 0.1  # dynamics simulation sampling rate
        sol = sp.integrate.solve_ivp(
            self.odes,
            (0, T_s + delta),
            y_t,
            t_eval=np.arange(0, T_s + delta, delta),
            args=(I, eps,),
        )  # dynamics simulation
        state = sol.y[:, -1]  # ODEs solution
        state = np.clip(state, 0.0, np.inf)  # clip to [0, inf)
        return state
