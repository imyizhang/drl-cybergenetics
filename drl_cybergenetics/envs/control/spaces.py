#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from typing import (
    TypeVar,
    Generic,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union
)

import numpy as np


T_cov = TypeVar('T_cov', covariant=True)


class Space(Generic[T_cov]):
    """Space that can be used to define valid action or observation space."""

    def __init__(
        self,
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[Type] = None,
    ) -> None:
        self._shape = shape if shape is None else tuple(shape)
        self.dtype = dtype
        self._rng = np.random.RandomState(seed=None)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self._shape

    @abc.abstractmethod
    def sample(self) -> T_cov:
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)


class Discrete(Space[int]):
    """Space that represents a finite subset of integers."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def sample(self) -> int:
        return self._rng.randint(0, self.n)


class Box(Space[np.ndarray]):
    """Space that represents a closed box in euclidean space."""

    def __init__(
        self,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: Optional[Sequence[int]] = None,
        dtype: Type = np.float32,
    ) -> None:
        if isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
            assert low.shape == high.shape
        if shape is not None:
            shape = tuple(shape)
        elif isinstance(low, np.ndarray):
            shape = low.shape
        elif isinstance(high, np.ndarray):
            shape = high.shape
        else:
            raise ValueError
        super().__init__(shape, dtype)
        # TODO: handle precision, bound check
        self.low = low.astype(self.dtype) if isinstance(low, np.ndarray) else np.full(shape, low, dtype=dtype)
        self.high = high.astype(self.dtype) if isinstance(high, np.ndarray) else np.full(shape, high, dtype=dtype)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(ow=self.low, high=self.high, size=self.shape).astype(self.dtype)
