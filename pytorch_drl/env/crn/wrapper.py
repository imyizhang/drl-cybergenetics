#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typing

import numpy as np

from .env import Env


class Wrapper(Env):
    """A wrapper of an environment to allow modular transformation."""

    def __init__(self, env: Env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action: typing.Union[int, np.ndarray], **kwargs):
        return self.env.step(action, **kwargs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed: typing.Optional[int] = None):
        return self.env.seed(seed=seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped
