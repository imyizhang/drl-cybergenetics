#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np


T_cov = typing.TypeVar('T_cov', covariant=True)

class Space(typing.Generic[T_cov]):

    def __init__(
        self,
        shape: typing.Optional[typing.Tuple[int, ...]] = None,
        dtype: typing.Optional[typing.Type] = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def sample(self) -> T_cov:
        raise NotImplementedError

    def seed(self, seed: typing.Optional[int] = None) -> None:
        self._rng.seed(seed)


class Discrete(Space[int]):

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def sample(self) -> int:
        return self._rng.randint(0, self.n)


class Box(Space[np.ndarray]):

    def __init__(
        self,
        low: typing.Union[float, np.ndarray],
        high: typing.Union[float, np.ndarray],
        shape: typing.Optional[typing.Tuple[int, ...]] = None,
        dtype: typing.Type = np.float32,
    ) -> None:
        if isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
            assert low.shape == high.shape
        if shape is None:
            if isinstance(low, np.ndarray):
                shape = low.shape
            elif isinstance(high, np.ndarray):
                shape = high.shape
            else:
                raise ValueError
        super().__init__(shape, dtype)
        self.low = low
        self.high = high

    def sample(self) -> np.ndarray:
        return self._rng.uniform(low=self.low, high=self.high, size=self.shape).astype(self.dtype)
