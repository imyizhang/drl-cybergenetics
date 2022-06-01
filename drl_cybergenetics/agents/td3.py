#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .ddpg import DDPGAgent


class TD3Agent(DDPGAgent):
    """ Twin Delayed DDPG (TD3).

    References:
        [1] 'Addressing Function Approximation Error in Actor-Critic Methods' (2018). arxiv.org/abs/1802.09477
        [2] spinningup.openai.com/en/latest/algorithms/td3.html
    """

    def __init__(
        self,
        device,
        buffer,
        actor,
        critic,
        learning_rate=1e-3,
        batch_size=64,
        discount_factor=0.99,
        exploration_steps=1000,
        exploration_noise=0.1,
        burnin_steps=64,
        update_every=10,
        grad_clipping=False,
        grad_limit=1.0,
        sync_coefficient=0.005,
        policy_delay=2,
        policy_noise=0.2,
        noise_limit=0.5,
    ):
        # Initialize replay buffer D and actor pi(s), critic Q(s, a)
        # Initialize target actor pi'(s), target critric Q'(s, a)
        # Hyperparameters for `act()`
        # Hyperparameters for `learn()`
        # Step
        super().__init__(
            device,
            buffer,
            actor,
            critic,
            learning_rate,
            batch_size,
            discount_factor,
            exploration_steps,
            exploration_noise,
            burnin_steps,
            update_every,
            grad_clipping,
            grad_limit,
            sync_coefficient,
        )
        # Extra hyperparameters for `learn()`
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_limit = noise_limit

    def learn(self):
        # Cache `burnin_steps` transitions before learning
        if self.curr_step < self.burnin_steps:
            return None
        # Learn every `update_every` steps
        if self.curr_step % self.update_every != 0:
            return None
        for step in range(self.update_every):
            # Sample random mini batch of transitions from replay buffer
            state, action, reward, next_state, done = self.recall()
            # Update critic
            critic_objective, critic_info = None, {}
            # Compute critic objective:
            # compute estimated Q(s, a);
            Q1, Q2 = self.critic.get_twin(state, action)
            critic_info.update(Q1=Q1, Q2=Q2)
            with torch.no_grad():
                # estimate a' ~ pi'(s') + epsilon, epsilon ~ clip(N(0, sigma), -c, c);
                next_action = self.actor_target.act(next_state, self.policy_noise,
                    noise_clipping=True, noise_limit=self.noise_limit)
                # estimate Q^{*}(s', a') ~ min_{i=1, 2} Q'_{i}(s', a');
                next_Q = torch.min(*self.critic_target.get_twin(next_state, next_action))
                # compute expected Q^{*}(s, a) = r(s, a) + gamma * max_{a'} Q(s', a');
                Q_target = reward + self.gamma * (1 - done) * next_Q
            # compute loss of temporal difference (TD) error
            critic_objective = self.critic_criterion(Q1, Q_target) + self.critic_criterion(Q2, Q_target)
            # Update critic by minimizing the critic objective
            self._update(self.critic_optimizer, critic_objective,
                grad_clipping=self.grad_clipping, nn=self.critic, grad_limit=self.grad_limit)
            # Update actor
            actor_objective, actor_info = None, {}
            # In TD3, actor is updated less frequently than critic is
            if step % self.policy_delay == 0:
                # Compute actor objective:
                self._freeze(self.critic)
                # estimate policy gradient ~ -E[Q(s, pi(s))]
                actor_objective = -self.critic(state, self.actor(state)).mean()
                self._unfreeze(self.critic)
                # Update actor by minimizing the actor objective
                self._update(self.actor_optimizer, actor_objective,
                    grad_clipping=self.grad_clipping, nn=self.actor, grad_limit=self.grad_limit)
                # Sync weights of target networks
                self._sync(self.critic_target, self.critic, soft_updating=True, tau=self.tau)
                self._sync(self.actor_target, self.actor, soft_updating=True, tau=self.tau)
        return actor_objective, actor_info, critic_objective, critic_info
