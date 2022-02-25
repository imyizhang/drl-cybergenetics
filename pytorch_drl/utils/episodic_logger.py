#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import pickle


class EpisodicLogger:

    def __init__(self):
        # for replay
        self.episodic_cache = []
        # for evaluation
        self.episodic_rewards = []
        self.episodic_aggregator_in_tolerance = []
        self.episodic_actor_objective = []
        self.episodic_critic_objective = []
        self.episodic_Q = []
        self.episodic_states = []
        self.episodic_actions = []
        # reset each episode
        self._reset_each_episode()

    def _reset_each_episode(self):
        # for evaluation
        self._rewards = []
        self._aggregator_in_tolerance = []
        self._actor_objective = []
        self._critic_objective = []
        self._Q = []
        self._states = []
        self._actions = []

    def step(self, state, action, reward, info, objectives):
        _state = state.view(-1).cpu().detach().numpy()
        _action = action.view(-1).cpu().detach().numpy()
        _reward = reward.cpu().detach().item()
        _count_in_tolerance = info['count_in_tolerance'] if 'count_in_tolerance' in info else 0
        _actor_objective = objectives['objective/actor'].cpu().detach().item() if (objectives is not None) and (objectives['objective/actor'] is not None) else 0
        _critic_objective = objectives['objective/critic'].cpu().detach().item() if objectives is not None else 0
        _Q = objectives['Q'].cpu().detach().item() if objectives is not None else 0
        # for evaluation
        self._rewards.append(_reward)
        self._aggregator_in_tolerance.append(_count_in_tolerance)
        self._actor_objective.append(_actor_objective)
        self._critic_objective.append(_critic_objective)
        self._Q.append(_Q)
        self._states.append(_state)
        self._actions.append(_action)

    def episode(self, env):
        # for relay
        _cache = copy.deepcopy(env._cache) if hasattr(env, '_cache') else None
        self.episodic_cache.append(_cache)
        # for evaluation
        self.episodic_rewards.append(self._rewards)
        self.episodic_aggregator_in_tolerance.append(self._aggregator_in_tolerance)
        self.episodic_actor_objective.append(self._actor_objective)
        self.episodic_critic_objective.append(self._critic_objective)
        self.episodic_Q.append(self._Q)
        self.episodic_states.append(self._states)
        self.episodic_actions.append(self._actions)
        # reset each episode
        self._reset_each_episode()

    def save(self, file):
        _logger = {
            #'cache': self.episodic_cache,
            'rewards': self.episodic_rewards,
            'tolerance': self.episodic_aggregator_in_tolerance,
            'actor_objective': self.episodic_actor_objective,
            'critic_objective': self.episodic_critic_objective,
            'Q': self.episodic_Q,
            'states': self.episodic_states,
            'actions': self.episodic_actions,
        }
        with open(file, 'wb') as f:
             pickle.dump(_logger, f)

    def load(self, file):
        with open(file, 'rb') as f:
             _logger = pickle.load(f)
        #self.episodic_cache = _logger['cache']
        self.episodic_rewards = _logger['rewards']
        self.episodic_aggregator_in_tolerance = _logger['tolerance']
        self.episodic_actor_objective = _logger['actor_objective']
        self.episodic_critic_objective = _logger['critic_objective']
        self.episodic_Q = _logger['Q']
        self.episodic_states = _logger['states']
        self.episodic_actions = _logger['actions']
