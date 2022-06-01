#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .dqn import DQNAgent


class DDQNAgent(DQNAgent):
    """Double DQN (DDQN) variant: Clipped Double Q-Learning.

    References:
        [1] 'Deep Reinforcement Learning with Double Q-learning' (2015). arxiv.org/abs/1509.06461
        [2] 'Addressing Function Approximation Error in Actor-Critic Methods' (2018). arxiv.org/abs/1802.09477
    """

    def act(self, state):
        with torch.no_grad():
            Q_distribution = self.critic(state)
            # Explore, Boltzmann exploration
            if torch.rand(1).item() < self.exploration_rate:
                action_probabilities = torch.nn.functional.softmax(Q_distribution, dim=1)
                action = self.actor.act(state, p=action_probabilities)
            # Exploit, a = pi^{*}(s) = argmax_{a} Q^{*}(s, a) ~ argmax_{a} Q(s, a)
            else:
                action = Q_distribution.max(dim=1, keepdim=True).indices
        # step
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
        Q1, Q2 = [Q.gather(dim=1, index=action) for Q in self.critic.get_twin(state)]
        with torch.no_grad():
            # estimate Q^{*}(s', a') ~ min_{1, 2} Q'(s', argmax_{a'} Q(s', a'));
            next_Q = torch.min(*self.critic_target.get_twin(next_state)).max(dim=1, keepdim=True).values
            # compute expected Q^{*}(s, a) = r(s, a) + gamma * max_{a'} Q(s', a');
            Q_target = reward + self.gamma * (1.0 - done) * next_Q
        # compute loss of temporal difference (TD) error
        critic_objective = self.critic_criterion(Q1, Q_target) + self.critic_criterion(Q2, Q_target)
        # Update critic by minimizing the critic objective
        self._update(self.critic_optimizer, critic_objective,
            grad_clipping=self.grad_clipping, grad_limit=self.grad_limit, nn=self.critic)
        return actor_objective, actor_info, critic_objective, critic_info
