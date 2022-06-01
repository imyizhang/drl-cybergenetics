#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import math


class ERScheduler(abc.ABC):

    def __init__(
        self,
        agent,
        decay_coefficient=0,
        end_er=0.05,
    ):
        if not hasattr(agent, 'exploration_rate'):
            raise RuntimeError
        self.agent = agent
        self.decay_coefficient = decay_coefficient
        self.start_er = getattr(agent, 'exploration_rate')
        self.end_er = end_er
        self.curr_er = self.start_er
        # intrinsic step counter
        self.curr_step = 0

    def __call__(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    def reset(self):
        self.curr_step = 0

    def step(self):
        # step
        self.curr_step += 1
        # update current exploration rate, self.curr_er, with intrinsic step
        self()
        # update self.agent.exploration_rate
        setattr(self.agent, 'exploration_rate', self.curr_er)


class LinearER(ERScheduler):

    def __call__(self):
        assert self.decay_coefficient > 1
        self.curr_er = self.end_er + (self.start_er - self.end_er) * (1.0 - self.curr_step / self.decay_coefficient)
        self.curr_er = max(self.end_er, self.curr_er)


class MultiplicativeER(ERScheduler):

    def __call__(self):
        assert self.decay_coefficient < 1 and self.decay_coefficient > 0
        self.curr_er = self.end_er + (self.start_er - self.end_er) * math.pow(self.decay_coefficient, self.curr_step)


class ExponentialER(ERScheduler):

    def __call__(self):
        assert self.decay_coefficient > 1
        self.curr_er = self.end_er + (self.start_er - self.end_er) * math.exp(-self.curr_step / self.decay_coefficient)
