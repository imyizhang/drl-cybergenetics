#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np
import scipy as sp
import scipy.signal as signal


class RefTrajectory(abc.ABC):
    """Time-varying reference trajectories to track."""

    ref_trajectory = None

    def __init__(self, scale: float, tolerance: float) -> None:
        self.scale = scale
        self.tolerance = tolerance

    @property
    def tolerance_margin(self) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        if self.ref_trajectory is None:
            return None
        return (
            self.ref_trajectory * (1 - self.tolerance),
            self.ref_trajectory * (1 + self.tolerance)
        )

    def __call__(self, t: np.ndarray):
        #assert t.ndim == 1
        #self.ref_trajectory = func(t)
        #return self.ref_trajectory, self.tolerance_margin
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ConstantRefTrajectory(RefTrajectory):
    """Constant reference trajectory."""

    def __init__(
        self,
        scale: float = 1.5,
        tolerance: float = 0.05,
    ) -> None:
        super().__init__(scale, tolerance)

    def __call__(self, t: np.ndarray):
        assert t.ndim == 1
        self.ref_trajectory = self.scale + np.zeros_like(t)  # dtype, the same as t
        return self.ref_trajectory, self.tolerance_margin


class SquareRefTrajectory(RefTrajectory):
    """Square-wave reference trajectory."""

    def __init__(
        self,
        scale: float = 1.5,
        tolerance: float = 0.05,
        period: float = 200.0,
        amplitude: float = 0.1,
        phase: float = 0.0,
    ) -> None:
        super().__init__(scale, tolerance)
        self.period = period
        self.amplitude = amplitude
        self.phase = phase

    def __call__(self, t: np.ndarray):
        assert t.ndim == 1
        self.ref_trajectory = self.scale + self.amplitude * signal.square(2 * np.pi * t / self.period + self.phase).astype(t.dtype)  # dtype, the same as t
        return self.ref_trajectory, self.tolerance_margin


class SineRefTrajectory(RefTrajectory):
    """Sine reference trajectory."""

    def __init__(
        self,
        scale: float = 1.5,
        tolerance: float = 0.05,
        period: float = 200.0,
        amplitude: float = 0.1,
        phase: float = 0.0,
    ) -> None:
        super().__init__(scale, tolerance)
        self.period = period
        self.amplitude = amplitude
        self.phase = phase

    def __call__(self, t: np.ndarray):
        assert t.ndim == 1
        self.ref_trajectory = self.scale + self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)  # dtype, the same as t
        return self.ref_trajectory, self.tolerance_margin


class BandPassFilter(RefTrajectory):
    """Band-pass filter (BPF) based switches to trigger."""

    def __init__(
        self,
        scale: float = 0.0,
        tolerance: float = 0.05,
        switches: np.ndarray = np.array([[200, 1.5]]),  # OFF-ON
    ) -> None:
        super().__init__(scale, tolerance)
        assert switches.ndim == 2 and switches.shape[1] == 2
        self.switches = switches

    def __call__(self, t: np.ndarray):
        assert t.ndim == 1
        self.ref_trajectory = self._signal(t)
        return self.ref_trajectory, self.tolerance_margin

    def _signal(self, t: np.ndarray) -> np.ndarray:
        y = np.zeros_like(t)  # dtype, the same as t
        mask_nan = True
        for i in range(self.switches.shape[0]):
            mask = (t == self.switches[i, 0])
            np.place(y, mask, self.switches[i, 1])
            mask_nan &= (1 - mask)
        np.place(y, mask_nan, np.nan)
        return y
