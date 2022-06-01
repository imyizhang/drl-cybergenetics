#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch

from .agent import Agent


class SACAgent(Agent):
    """Soft Actor-Critic (SAC).

    References:
        [1] 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor' (2018). arxiv.org/abs/1801.01290
        [2] spinningup.openai.com/en/latest/algorithms/sac.html
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
        burnin_steps=64,
        update_every=10,
        grad_clipping=False,
        grad_limit=1.0,
        sync_coeff=0.005,
        regularization_coeff=0.2,
    ):
        # Initialize replay buffer D and actor pi(s), critic Q(s, a)
        super().__init__(device, actor, critic, buffer, actor_lr, critic_lr, batch_size, discount)
        # Initialize target critric Q'(s, a)
        self.critic_target = copy.deepcopy(self.critic)
        # Hyperparameters for `act()`
        self.exploration_steps = exploration_steps
        self.stochastic_exploration = True
        # Hyperparameters for `learn()`
        self.burnin_steps = burnin_steps
        self.update_every = update_every
        self.grad_clipping = grad_clipping
        self.grad_limit = grad_limit
        self.tau = sync_coeff
        self.alpha = regularization_coeff
        # Step
        self.curr_step = 0

    def train(self):
        self.actor.train()
        self.critic.train()
        self.critic_target.eval()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()
        self.stochastic_exploration = False

    def act(self, state):
        # Trick: improve exploration at the beginning
        if self.stochastic_exploration and self.curr_step < self.exploration_steps:
            action = self.actor.explore(state)
        # Normal SAC exploration, a = pi^{*}(s) ~ pi(· | s)
        else:
            action = self.actor.act(state, self.stochastic_exploration)
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
            # compute estimated Q_{i}(s, a), i=1, 2;
            Q1, Q2 = self.critic.get_twin(state, action)
            critic_info.update(Q1=Q1, Q2=Q2)
            with torch.no_grad():
                # estimate a' ~ pi(· | s');
                next_action, next_log_prob = self.actor(next_state)
                # estimate Q^{*}(s', a') ~ min_{i=1, 2} Q'_{i}(s', a');
                next_Q = torch.min(*self.critic_target.get_twin(next_state, next_action))
                # compute expected Q^{*}(s, a) = r(s, a) + gamma * [max_{a'} Q(s', a') - alpha * log pi(a' | s')];
                Q_target = reward + self.gamma * (1 - done) * (next_Q - self.alpha * next_log_prob)
            # compute loss of temporal difference (TD) error
            critic_objective = self.critic_criterion(Q1, Q_target) + self.critic_criterion(Q2, Q_target)
            # Update critic by minimizing the critic objective
            self._update(self.critic_optimizer, critic_objective,
                grad_clipping=self.grad_clipping, nn=self.critic, grad_limit=self.grad_limit)
            # Update actor
            actor_objective, actor_info = None, {}
            # Compute actor objective:
            self._freeze(self.critic)
            # estimate a ~ pi(· | s);
            action, log_prob = self.actor(state)
            actor_info.update(log_prob=log_prob)
            # estimate policy gradient ~ -E[min_{i=1, 2} Q_{i}(s, a) - alpha * log pi(a | s)]
            actor_objective = -(torch.min(*self.critic.get_twin(state, action)) - self.alpha * log_prob).mean()
            self._unfreeze(self.critic)
            # Update actor by minimizing the actor objective
            self._update(self.actor_optimizer, actor_objective,
                grad_clipping=self.grad_clipping, nn=self.actor, grad_limit=self.grad_limit)
            # Sync weights of target networks
            self._sync(self.critic_target, self.critic, soft_updating=True, tau=self.tau)
        return actor_objective, actor_info, critic_objective, critic_info
