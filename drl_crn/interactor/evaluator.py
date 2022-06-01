#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import itertools


def evaluate(
    agent,
    env,
    logger,
    episodes=10,
    max_steps_per_episode=1000,
):
    # Set agent in evaluation mode
    agent.eval()
    # For each episode
    for episode in range(episodes):
        # Initialize env and state
        state = env.reset()
        # For each step
        for step in itertools.count():
            # Select an action
            action = agent.act(state)
            # Perform the action and observe new state from env
            next_state, reward, done, info = env.step(action)
            # Update state
            state = next_state
            # Handle episode end
            # TODO: logging `episodic_accumulative_reward`, `episodic_duration`
            if done or (step + 1 == max_steps_per_episode):
                break
    # Close env
    env.close()


class Evaluator(abc.ABC):

    def __init__(self, agent, env, logger):
        self.agent = agent
        self.env = env
        self.logger = logger

    def __call__(self, **kwargs):
        return evaluate(self.agent, self.env, self.logger, **kwargs)
