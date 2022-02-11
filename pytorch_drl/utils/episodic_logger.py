#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

import matplotlib.pyplot as plt


class Cache(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'Cache' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value


class EpisodicLogger:

    def __init__(self):
        # for replay
        self.episodic_cache = []
        self.episodic_actions = []
        self.episodic_actions_taken = []
        self.episodic_trajectory = []
        self.episodic_observations = []
        self.episodic_rewards = []
        self.episodic_duration = []
        # for evaluation
        self.episodic_counts_in_tolerance = []
        self.episodic_loss = []
        self.episodic_Q = []
        self.episodic_return = []
        # initialize
        self._init()

    @property
    def cache(self):
        return self.episodic_cache

    @property
    def actions(self):
        return self.episodic_actions

    @property
    def actions_taken(self):
        return self.episodic_actions_taken

    @property
    def trajectories(self):
        return self.episodic_trajectory

    @property
    def observations(self):
        return self.episodic_observations

    @property
    def rewards(self):
        return self.episodic_rewards

    @property
    def durations(self):
        return self.episodic_duration

    @property
    def tolerance(self):
        return self.episodic_counts_in_tolerance

    @property
    def loss(self):
        return self.episodic_loss

    @property
    def Q(self):
        return self.episodic_Q

    def _init(self):
        # for replay
        self._actions = []
        self._actions_taken = []
        self._trajectory = []
        self._observations = []
        self._rewards = []
        self._steps_done = 0
        # for evaluation
        self._counts_in_tolerance = []
        self._loss = []
        self._Q = []

    def step(self, reward, info, losses):
        _reward = reward.cpu().detach().item()
        _loss = losses['loss/critic'].cpu().detach().item() if losses is not None else 0
        _Q = losses['Q'].cpu().detach().item() if losses is not None else 0
        # for replay
        self._actions.append(info['action'])
        self._actions_taken.append(info['action_taken'])
        self._trajectory.append(info['state'])
        self._observations.append(info['observation'])
        self._rewards.append(_reward)
        self._steps_done += 1
        # for evaluation
        self._counts_in_tolerance.append(info['count_in_tolerance'])
        self._loss.append(_loss)
        self._Q.append(_Q)

    def episode(self):
        # for relay
        self.episodic_actions.append(self._actions)
        self.episodic_actions_taken.append(self._actions_taken)
        self.episodic_trajectory.append(self._trajectory)
        self.episodic_observations.append(self._observations)
        self.episodic_rewards.append(self._rewards)
        self.episodic_duration.append(self._steps_done)
        cache = Cache()
        cache.trajectory = self._trajectory
        cache.observations = self._observations
        cache.actions = self._actions
        cache.actions_taken = self._actions_taken
        cache.rewards = self._rewards
        cache.steps_done = self._steps_done
        self.episodic_cache.append(cache)
        # for evaluation
        self.episodic_counts_in_tolerance.append(self._counts_in_tolerance)
        self.episodic_loss.append(self._loss)
        self.episodic_Q.append(self._Q)
        # initialize
        self._init()

    def save(self, file):
        _logger = {
            'cache': self.cache,
            'reward':  self.reward,
            'tolerance': self.tolerance,
            'loss': self.loss,
            'Q': self.Q,
        }
        with open(file, 'wb') as f:
             pickle.dump(_logger, f)

    def load(self, file):
        with open(file, 'rb') as f:
             _logger = pickle.load(f)
        self.episodic_cache = _logger['cache']
        # for evaluation
        self.episodic_reward = _logger['reward']
        self.episodic_aggregator_in_tolerance = _logger['tolerance']
        self.episodic_loss = _logger['loss']
        self.episodic_Q = _logger['Q']
        return self

    def plot(self):
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(6, 5),
            gridspec_kw={'height_ratios': [1, 1]}
        )
        #fig.tight_layout()
        axs[0].plot([sum(r) for r in self.rewards], marker='.', color='tab:orange')
        axs[0].set_ylabel('reward')
        axs[0].grid(True)
        axs[1].plot([sum(t) / len(t) * 100 for t in self.tolerance], marker='.', color='tab:red')
        axs[1].set_ylim([0, 100])
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('% states in tolerance zone')
        axs[1].grid(True)
        plt.show()
