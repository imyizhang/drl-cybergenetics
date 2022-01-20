#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .agent import Agent
from ..nn import DummyActor


class DummyAgent(Agent):

    def __init__(
        self,
        device,
        actor,
        critic=None,
        discount_factor=None,
        learning_rate=None,
        buffer_capacity=None,
        batch_size=None,
    ):
        super().__init__(
            device,
            actor,
            critic,
            discount_factor,
            learning_rate,
            buffer_capacity,
            batch_size,
        )
        assert isinstance(self.actor, DummyActor)
        # step counter
        self.curr_step = 0

    def act(self, state):
        # explore
        action = self.actor.act(state)
        # step
        self.curr_step += 1
        return action

    def train(self):
        return None

    def learn(self):
        return None
