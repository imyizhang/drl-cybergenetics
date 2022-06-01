#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .vpg import VPGAgent


class A2CAgent(VPGAgent):
    """Advantage Actor Critic (A2C).

    References:
        [1] 'Asynchronous Methods for Deep Reinforcement Learning' (2016). arxiv.org/abs/1602.01783
    """

    def __init__(
        self,
        device,
        actor,
        critic,
        buffer,
        actor_lr=5e-4,
        critic_lr=1e-3,
        batch_size=64,
        discount=0.99,
        gae_lambda=0.95,
        actor_update_iters=100,
        critic_update_iters=100,
        grad_clipping=False,
        grad_limit=1.0,
        regularization_coeff=0.2,
    ):
        # Initialize rollout buffer D and actor pi(s), critic V(s)
        super().__init__(
            device,
            actor,
            critic,
            buffer,
            actor_lr,
            critic_lr,
            batch_size,
            discount,
            gae_lambda,
            actor_update_iters,
            critic_update_iters,
            grad_clipping,
            grad_limit,
        )
        self.alpha = regularization_coeff

    def learn(self):
        # Sample random mini batch of transitions from rollout buffer
        state, action, reward_to_go, advantage, log_prob = self.recall()
        # Update actor
        actor_objective, actor_info = None, {}
        for _ in range(self.update_iters):
            # Compute actor objective:
            entropy = self.actor.entropy(self.actor(state))
            actor_objective = -(log_prob * advantage - self.alpha * entropy).mean()
            # Update actor by minimizing actor objective
            self._update(self.actor_optimizer, actor_objective,
                grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.actor)
        # Update critic
        critic_objective, critic_info = None, {}
        for _ in range(self.update_iters):
            # Compute critic objective: -E[(V(s) - reward_to_go) ** 2]
            critic_objective = ((self.critic(state) - reward_to_go) ** 2).mean()
            # Update critic by minimizing critic objective
            self._update(self.critic_optimizer, critic_objective,
                grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.critic)
        return actor_objective, actor_info, critic_objective, critic_info
