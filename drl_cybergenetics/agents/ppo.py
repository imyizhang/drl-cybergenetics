#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .vpg import VPGAgent


class PPOAgent(VPGAgent):
    """Proximal Policy Optimization (PPO) variant: PPO-Clip.

    References:
        [1] 'High-Dimensional Continuous Control Using Generalized Advantage Estimation' (2015). arxiv.org/abs/1506.02438
        [2] 'Proximal Policy Optimization Algorithms' (2017). arxiv.org/abs/1707.06347
        [3] spinningup.openai.com/en/latest/algorithms/ppo.html
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
        ratio_clip=0.1,
        kl_threshold=0.01,
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
        self.ratio_clip = ratio_clip
        self.kl_threshold = kl_threshold

    def learn(self):
        # Sample random batch of transitions from rollout buffer
        state, action, reward_to_go, advantage, log_prob = self.recall()
        # Trick: implement advantage batch normalization
        # TODO: remove effect from padding
        advantage = (advantage - advantage.mean()) / advantage.std()
        # Update actor
        actor_objective, actor_info = None, {}
        for _ in range(self.actor_update_iters):
            # Compute actor objective:
            pi_distributio = self.actor(state)
            curr_log_prob = self.actor.log_prob(pi_distributio, action)
            ratio = torch.exp(curr_log_prob - log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip)
            actor_objective = -(torch.min(ratio * advantage, clipped_ratio * advantage)).mean()
            # Update actor by minimizing actor objective
            self._update(self.actor_optimizer, actor_objective,
                grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.actor)
            # Trick: early stopping, ensuring reasonable policy updates
            kl = (log_prob - curr_log_prob).mean()
            if kl >= 1.5 * self.kl_threshold:
                break
        # Update critic
        critic_objective, critic_info = None, {}
        for _ in range(self.critic_update_iters):
            # Compute critic objective: -E[(V(s) - reward_to_go) ** 2]
            critic_objective = ((self.critic(state) - reward_to_go) ** 2).mean()
            # Update critic by minimizing critic objective
            self._update(self.critic_optimizer, critic_objective,
                grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.critic)
        return actor_objective, actor_info, critic_objective, critic_info
