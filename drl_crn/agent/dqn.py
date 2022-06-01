#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch

from .agent import Agent


class DQNAgent(Agent):
    """Deep Q Network (DQN).

    References:
        [1] 'Human-Level Control Through Deep Reinforcement Learning' (2015). nature.com/articles/nature14236
        [2] 'Deep Recurrent Q-Learning for Partially Observable MDPs' (2015). arxiv.org/abs/1507.06527
    """

    def __init__(
        self,
        device,
        actor,
        critic,
        buffer,
        actor_lr=1e-3,
        critic_lr=1e-3,
        batch_size=64,
        discount=0.99,
        exploration_rate=0.9,
        burnin_steps=64,
        update_every=1,
        grad_clipping=False,
        grad_limit=1.0,
        sync_every=4000,
    ):
        # Initialize replay buffer D and critic Q(s, a)
        super().__init__(device, actor, critic, buffer, actor_lr, critic_lr, batch_size, discount)
        # Initialize target critric Q'(s, a)
        self.critic_target = copy.deepcopy(self.critic)
        # Hyperparameters for `act()`
        self.exploration_rate = exploration_rate
        # Hyperparameters for `learn()`
        self.burnin_steps = burnin_steps
        self.update_every = update_every
        self.grad_clipping = grad_clipping
        self.grad_limit = grad_limit
        self.sync_every = sync_every
        # Step
        self.curr_step = 0

    def train(self):
        self.critic.train()
        self.critic_target.eval()

    def eval(self):
        self.critic.eval()
        self.critic_target.eval()
        self.exploration_rate = 0.0

    def act(self, state):
        # Explore, epsilon-greedy policy
        if torch.rand(1).item() < self.exploration_rate:
            action = self.actor.act(state)
        # Exploit, a = pi^{*}(s) = argmax_{a} Q^{*}(s, a) ~ argmax_{a} Q(s, a)
        else:
            with torch.no_grad():
                action = self.critic(state).max(dim=1, keepdim=True).indices
        # Step
        self.curr_step += 1
        return action

    def learn(self):
        # Cache `burnin_steps` transitions before learning
        if self.curr_step < self.burnin_steps:
            return None
        # Learn every `update_every` steps
        if self.curr_step % self.update_every != 0:
            return None
        # Sync weights of target networks every `sync_every` steps
        if self.curr_step % self.sync_every == 0:
            self._sync(self.critic_target, self.critic)
        # Sample random mini batch of transitions from replay buffer
        state, action, reward, next_state, done = self.recall()
        # Update actor
        actor_objective, actor_info = None, {}
        # Update critic
        critic_objective, critic_info = None, {}
        # Compute critic objective:
        # compute estimated Q(s, a);
        Q = self.critic(state).gather(dim=1, index=action)
        critic_info['Q'] = Q.mean()
        with torch.no_grad():
            # estimate Q^{*}(s', a') ~ max_{a'} Q'(s', a');
            next_Q = self.critic_target(next_state).max(dim=1, keepdim=True).values
            # double Q-learning: estimate Q^{*}(s', a') ~ Q'(s', argmax_{a'} Q(s', a'));
            # next_Q = self.critic_target(next_state).gather(1, self.critic(next_state).max(dim=1, keepdim=True).indices)
            # compute expected Q^{*}(s, a) = r(s, a) + gamma * max_{a'} Q(s', a');
            Q_target = reward + self.gamma * (1.0 - done) * next_Q
        # compute loss of temporal difference (TD) error
        critic_objective = self.critic_criterion(Q, Q_target)
        # Update critic by minimizing the critic objective
        self._update(self.critic_optimizer, critic_objective,
            grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.critic)
        return actor_objective, actor_info, critic_objective, critic_info
