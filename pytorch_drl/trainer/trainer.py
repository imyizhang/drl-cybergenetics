#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import itertools

from ..utils import EpisodicLogger, replay


class Trainer(abc.ABC):

    def __init__(self, env, agent, n_episodes):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes

    def __call__(self):
        raise NotImplementedError


class OffPolicyTrainer(Trainer):

    def __init__(
        self,
        env,
        agent,
        n_episodes=10,
        bs_scheduler=None,
        er_scheduler=None,
    ):
        super().__init__(env, agent, n_episodes)
        self.bs_scheduler = bs_scheduler
        self.er_scheduler = er_scheduler
        # logger
        self.logger = EpisodicLogger()
        # reward func for training
        self.reward_func = None

    def __call__(self, reward_func):
        self.reward_func = reward_func
        # set the agent in training mode
        self.agent.train()
        for episode in range(self.n_episodes):
            # initialize the env and state
            state = self.env.reset()
            # episode training
            for step in itertools.count():
                # select an action
                action = self.agent.act(state)
                # exploration rate decay for agents in DQN family
                if self.er_scheduler is not None:
                    self.er_scheduler.step()
                # perform the action and observe new state
                next_state, reward, done, info = self.env.step(action, reward_func=self.reward_func)
                # buffer the experience
                if self.agent.buffer is not None:
                    self.agent.cache(state, action, reward, done, next_state)
                # batch_size update for learning
                if self.bs_scheduler is not None:
                    self.bs_scheduler.step()
                # learn from the experience
                objectives = self.agent.learn()
                # step logging
                self.logger.step(state, action, reward, info, objectives)
                # update the state
                state = next_state
                # check if end
                if done:
                    break
            # episode logging
            self.logger.episode(self.env)
        # close the env
        self.env.close()
        return self.logger

    def evaluate(self):
        # initialize the env
        state = self.env.reset()
        # episode training
        for step in itertools.count():
            # select an action
            action = self.agent.act(state)
            # perform the action and observe new state
            next_state, reward, done, info = self.env.step(action, reward_func=self.reward_func)
            # step logging
            self.logger.step(state, action, reward, info, None)
            # update the state
            state = next_state
            # check if end
            if done:
                break
        # episode logging
        self.logger.episode(self.env)
        # close the env
        self.env.close()
        # replay
        return replay(self.env, self.logger, episode=-1)
