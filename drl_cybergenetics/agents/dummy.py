#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .agent import Agent


class DummyAgent(Agent):

    def __init__(
        self,
        device,
        actor,
        critic=None,
        buffer=None,
        actor_lr=None,
        critic_lr=None,
        batch_size=None,
        discount=None,
    ):
        super().__init__(device, actor, critic, buffer, actor_lr, critic_lr, batch_size, discount)
        # Step
        self.curr_step = 0

    def train(self):
        pass

    def eval(self):
        pass

    def act(self, state):
        # Exploit
        action = self.actor.act(state)
        # Step
        self.curr_step += 1
        return action

    def learn(self):
        return None
