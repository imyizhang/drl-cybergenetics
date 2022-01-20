#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import typing

import numpy as np


class Env(abc.ABC):
    """An environment with arbitrary behind-the-scenes dynamics, which can be
    partially or fully observed.

    OpenAI Gym style API methods:

        reset
        step
        render
        close
        seed

    """

    def __init__(self):
        self._rng = np.random.RandomState(seed=None)

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: typing.Union[int, np.ndarray]):
        raise NotImplementedError

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed: typing.Optional[int] = None):
        self._rng.seed(seed)

    @property
    def unwrapped(self):
        return self
