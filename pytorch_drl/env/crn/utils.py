#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np


class RefTrajectory(abc.ABC):
    """Time-varying reference trajectories to track."""

    def __init__(
        self,
        scale: float = 1.5,
        tolerance: float = 0.05,
    ) -> None:
        self.scale = scale
        self.tolerance = tolerance
        self._ref_trajectory = None

    @property
    def ref_trajectory(self) -> typing.Optional[np.ndarray]:
        return self._ref_trajectory

    @property
    def tolerance_margin(self) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        if self.ref_trajectory is None:
            return None
        return (
            self._ref_trajectory * (1 - self.tolerance),
            self._ref_trajectory * (1 + self.tolerance)
        )

    def __call__(self, t: np.ndarray):
        #assert t.ndim == 1
        #self._ref_trajectory =
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
        self._ref_trajectory = self.scale + np.zeros(t.shape)
        return self.ref_trajectory, self.tolerance_margin


class StepRefTrajectory(RefTrajectory):
    """Two-step reference trajectory."""

    def __init__(
        self,
        scale: float = 1.5,
        tolerance: float = 0.05,
        step: float = 200.0,
        gamma: float = 1.1,
    ) -> None:
        super().__init__(scale, tolerance)
        self.step = step
        self.gamma = gamma

    def __call__(self, t: np.ndarray):
        assert t.ndim == 1
        self._ref_trajectory = self.scale * (t <= self.step) + self.scale * self.gamma * (t > self.step)
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
        self._ref_trajectory = self.scale + self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)
        return self.ref_trajectory, self.tolerance_margin
