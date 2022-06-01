#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

import torch

from .agent import Agent


def _discounted_cumsum(l, discount):
    def _add(total, value):
        return discount * total + value
    return list(itertools.accumulate(l[::-1], func=_add, initial=0))[::-1][:-1]


class VPGAgent(Agent):
    """Vanilla Policy Gradient (VPG).

    References:
        [1] 'High-Dimensional Continuous Control Using Generalized Advantage Estimation' (2017). arxiv.org/abs/1506.02438
        [2] spinningup.openai.com/en/latest/algorithms/vpg.html
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
        actor_update_iters=1,
        critic_update_iters=100,
        grad_clipping=False,
        grad_limit=1.0,
    ):
        # Initialize rollout buffer D and actor pi(s), critic V(s)
        super().__init__(device, actor, critic, buffer, actor_lr, critic_lr, batch_size, discount)
        self.gae_lambda = gae_lambda
        # Hyperparameters for `act()`
        # Hyperparameters for `learn()`
        self.actor_update_iters = actor_update_iters
        self.critic_update_iters = critic_update_iters
        self.grad_clipping = grad_clipping
        self.grad_limit = grad_limit
        # Step
        self.curr_step = 0

    def recall(self):
        # Update trajectories
        updated_trajectories = []
        # For each trajectory
        for t in self.buffer.trajectories:
            # Bootstrap
            with torch.no_grad():
                last_V = (1.0 - t.done[-1]) * self.critic(t.next_state[-1])
            reward = list(t.reward) + [last_V]
            V = list(t.V) + [last_V]
            # Compute reward-to-go
            # G_{t} := sum_{i=t}^{T} gamma^{i-t} r_{i}
            reward_to_go = _discounted_cumsum(reward, self.gamma)[:-1]
            # Estimate A via GAE(gamma, lambda):
            # compute TD errors, delta_{t} := r_{t} + gamma V(s_{t+1}) - V(s_{t});
            delta = reward[:-1] + self.gamma * V[1:] - V[:-1]
            # estimate A, A_{t}^{GAE} := sum_{i=t}^{T} (gamma lambda)^{i-t} delta_{i}
            advantage = _discounted_cumsum(delta, self.gamma * self.gae_lambda)
            updated_trajectories.append((t.state, t.action, reward_to_go, advantage, t.log_prob))
        self.buffer.trajectories = updated_trajectories
        batch = self.buffer.pop_all()
        return tuple(*[self._to(tensor, self.device) for tensor in self._as_tensors(batch)])

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def step(self, state):
        action, log_prob = self.actor.step(state)
        with torch.no_grad():
            V = self.critic(state)
        # Step
        self.curr_step += 1
        return action, V, log_prob

    def act(self, state):
        # Explore
        action = self.actor.act(state)
        # Step
        self.curr_step += 1
        return action

    def learn(self):
        # Sample random batch of transitions from rollout buffer
        state, action, reward_to_go, advantage, log_prob = self.recall()
        # Trick: implement advantage batch normalization
        # TODO: remove effect from padding
        advantage = (advantage - advantage.mean()) / advantage.std()
        # Update actor
        actor_objective, actor_info = None, {}
        for _ in range(self.actor_update_iters):
            # Compute actor objective: -E[log pi(a | s) * A(s, a)]
            actor_objective = - (self.actor.log_prob(self.actor(state), action) * advantage).mean()
            # Update actor by minimizing actor objective
            self._update(self.actor_optimizer, actor_objective,
                grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.actor)
        # Update critic
        critic_objective, critic_info = None, {}
        for _ in range(self.critic_update_iters):
            # Compute critic objective: -E[(V(s) - reward-to-go) ** 2]
            critic_objective = ((self.critic(state) - reward_to_go) ** 2).mean()
            # Update critic by minimizing critic objective
            self._update(self.critic_optimizer, critic_objective,
                grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.critic)
        return actor_objective, actor_info, critic_objective, critic_info
