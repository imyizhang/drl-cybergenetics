#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

from .trainer import Trainer
from ..utils import EpisodicLogger


class DummyTrainer(Trainer):

    def __init__(
        self,
        env,
        agent,
        n_episodes=1,
        n_timesteps=None,
    ):
        super().__init__(
            env,
            agent,
            n_episodes,
            n_timesteps
        )
        self.logger = EpisodicLogger()

    def __call__(self, reward_func):
        # set the agent in training mode
        self.agent.train()
        for episode in range(self.n_episodes):
            # initialize the env and state
            state = self.env.reset()
            # initialize the logger
            self.logger.reset()
            # episode training
            timesteps = itertools.count() if self.n_timesteps is None else range(self.n_timesteps)
            for step in timesteps:
                # select an action
                action = self.agent.act(state)
                # perform the action and observe new state
                next_state, reward, done, info = self.env.step(action, reward_func=reward_func)
                # buffer the experience
                #self.agent.cache(state, action, reward, done, next_state)
                # learn from the experience
                losses = self.agent.learn()
                # update the state
                state = next_state
                # step logging
                self.logger.step(self.env, action, state, reward, info, losses)
                # check if end
                #if done
                #    break
            # episode logging
            self.logger.episode()
        self.env.close()
        return self.logger
