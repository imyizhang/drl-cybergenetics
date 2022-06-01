#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typing

import numpy as np

from .env import Env


class Wrapper(Env):
    """A wrapper of an environment to allow modular transformation."""

    def __init__(self, env: Env) -> None:
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def step(self, action: typing.Union[int, np.ndarray], **kwargs):
        return self.env.step(action, **kwargs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self) -> None:
        return self.env.close()

    def seed(self, seed: typing.Optional[int] = None) -> None:
        return self.env.seed(seed=seed)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped
