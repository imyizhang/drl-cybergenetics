#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc


class Trainer(abc.ABC):

    def __init__(
        self,
        env,
        agent,
        n_episodes,
        n_timesteps,
    ):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.n_timesteps = n_timesteps

    def __call__(self):
        raise NotImplementedError
