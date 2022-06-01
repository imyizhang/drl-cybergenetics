#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch

from .agent import Agent


class DDPGAgent(Agent):
    """Deep Deterministic Policy grad (DDPG).

    References:
        [1] 'Continuous Control with Deep Reinforcement Learning' (2015). arxiv.org/pdf/1509.02971
        [2] spinningup.openai.com/en/latest/algorithms/ddpg.html
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
        exploration_steps=1000,
        exploration_noise=0.1,
        burnin_steps=64,
        update_every=10,
        grad_clipping=False,
        grad_limit=1.0,
        sync_coeff=0.005,
    ):
        # Initialize replay buffer D and actor pi(s), critic Q(s, a)
        super().__init__(device, actor, critic, buffer, actor_lr, critic_lr, batch_size, discount)
        # Initialize target actor pi'(s), target critric Q'(s, a)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        # Hyperparameters for `act()`
        self.exploration_steps = exploration_steps
        self.exploration_noise = exploration_noise
        # Hyperparameters for `learn()`
        self.burnin_steps = burnin_steps
        self.update_every = update_every
        self.grad_clipping = grad_clipping
        self.grad_limit = grad_limit
        self.tau = sync_coeff
        # Step
        self.curr_step = 0

    def train(self):
        self.actor.train()
        self.actor_target.eval()
        self.critic.train()
        self.critic_target.eval()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
        self.exploration_noise = 0.0

    def act(self, state):
        # Trick: improve exploration at the beginning
        if self.exploration_noise != 0.0 and self.curr_step < self.exploration_steps:
            action = self.actor.explore(state)
        # Normal DDPG exploration, a = pi^{*}(s) ~ pi(s) + epsilon, epsilon ~ N(0, sigma)
        else:
            action = self.actor.act(state, self.exploration_noise)
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
        for _ in range(self.update_every):
            # Sample random mini batch of transitions from replay buffer
            state, action, reward, next_state, done = self.recall()
            # Update critic
            critic_objective, critic_info = None, {}
            # Compute critic objective:
            # compute estimated Q(s, a);
            Q = self.critic(state, action)
            critic_info.update(Q=Q)
            with torch.no_grad():
                # estimate Q^{*}(s', a') ~ Q'(s', pi'(s'));
                next_Q = self.critic_target(next_state, self.actor_target(next_state))
                # compute expected Q^{*}(s, a) = r(s, a) + gamma * max_{a'} Q(s', a');
                Q_target = reward + self.gamma * (1.0 - done) * next_Q
            # compute loss of temporal difference (TD) error
            critic_objective = self.critic_criterion(Q, Q_target)
            # Update critic by minimizing critic objective
            self._update(self.critic_optimizer, critic_objective,
                grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.critic)
            # Update actor
            actor_objective, actor_info = None, {}
            # Compute actor objective:
            self._freeze(self.critic)
            # estimate policy gradient ~ -E[Q(s, pi(s))]
            actor_objective = -self.critic(state, self.actor(state)).mean()
            self._unfreeze(self.critic)
            # Update actor by minimizing actor objective
            self._update(self.actor_optimizer, actor_objective,
                grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.actor)
            # Sync weights of target networks
            self._sync(self.critic_target, self.critic, soft_updating=True, tau=self.tau)
            self._sync(self.actor_target, self.actor, soft_updating=True, tau=self.tau)
        return actor_objective, actor_info, critic_objective, critic_info
