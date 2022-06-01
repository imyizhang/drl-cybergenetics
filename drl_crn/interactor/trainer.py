#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

from .evaluator import evaluate


def train_offpolicy(
    agent,
    env,
    logger,
    epochs=100,
    steps_per_epoch=1000,
    max_steps_per_episode=1000,
    exploration_scheduler=None,
    batch_scheduler=None,
    save_every=1,
    eval_every=1,
    **eval_kwargs,
):
    # Set agent in training mode
    agent.train()
    # Initialize env and state
    state = env.reset()
    # For each step
    for step in range(steps_per_epoch * epochs):
        # Collect experience via interacting with env:
        # select an action from any policy;
        action = agent.act(state)
        # exploration rate decay for agent acting;
        if exploration_scheduler is not None:
            exploration_scheduler.step()
        # perform the action and observe new state from env;
        next_state, reward, done, info = env.step(action)
        # ignore done signal if it comes from hitting time horizon;
        done = False if step + 1 == max_steps_per_episode else done
        # cache transition to replay buffer;
        agent.cache(state, action, reward, next_state, done)
        # update state;
        state = next_state
        # handle episode end;
        # TODO: logging `episodic_accumulative_reward`, `episodic_duration`
        if done or (step + 1 == max_steps_per_episode):
            state = env.reset()
        # Learn from experience
        feedback = agent.learn()
        # TODO: logging feedback: `actor_info`, `actor_objective`, `critic_info`, `critic_objective`
        # batch size update for agent learning
        if batch_scheduler is not None:
            batch_scheduler.step()
        # Handle epoch end
        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            if (epoch % save_every == 0) or (epoch == epochs - 1):
                agent.save()
            if (epoch % eval_every == 0) or (epoch == epochs - 1):
                evaluate(agent, env, logger, **eval_kwargs)
            # TODO: logging data analysis
    # Close env
    env.close()


def train_onpolicy(
    agent,
    env,
    logger,
    epochs=100,
    steps_per_epoch=1000,
    max_steps_per_episode=1000,
    save_every=10,
    eval_every=10,
    **eval_kwargs,
):
    # Set agent in training mode
    agent.train()
    # Initialize env and state
    state = env.reset()
    # For each epoch
    for epoch in range(epochs):
        # Collect experience via interacting with env:
        for step in range(steps_per_epoch):
            # select an action from a policy;
            action, V, log_prob = agent.step(state)
            # perform the action and observe new state from env;
            next_state, reward, done, info = env.step(action)
            # cache transition to rollout buffer;
            agent.cache(state, action, reward, V, log_prob, done)
            # update state;
            state = next_state
            # handle trajectory end
            timeout = (step + 1 == max_steps_per_episode)
            epoch_ended = (step + 1 == steps_per_epoch)
            if done or timeout or epoch_ended:
                # bootstrap
                _, V, _ = agent.step(state)
                (1 - done) * V
                state = env.reset()
        # Learn from experience via updating above policy
        feedback = agent.learn()
        # Handle epoch end
        if (epoch % save_every == 0) or (epoch == epochs - 1):
            agent.save()
        if (epoch % eval_every == 0) or (epoch == epochs - 1):
            evaluate(agent, env, logger, **eval_kwargs)
    # Close env
    env.close()


class Trainer(abc.ABC):

    def __init__(self, agent, env, logger):
        self.agent = agent
        self.env = env
        self.logger = logger

    def __call__(self, **kwargs):
        raise NotImplementedError


class OffPolicyTrainer(Trainer):

    def __call__(self, **kwargs):
        return train_offpolicy(self.agent, self.env, self.logger, **kwargs)


class OnPolicyTrainer(Trainer):

    def __call__(self, **kwargs):
        return train_onpolicy(self.agent, self.env, self.logger, **kwargs)
