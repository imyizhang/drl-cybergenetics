#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import collections
import random

from .segment_tree import SumSegmentTree, MinSegmentTree


Experience = collections.namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'done', 'next_state')
)


class Transition(Experience):
    pass


class Episode(Experience):
    pass


class ReplayBuffer(abc.ABC):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory.clear()

    @abc.abstractmethod
    def push(self, *experience):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    def seed(seed=None):
        random.seed(seed)


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, alpha):
        super().__init__(capacity)
        self.max_priority = 1.0
        self.alpha = alpha
        self._sum_tree = SumSegmentTree(maxlen=capacity)
        self._min_tree = MinSegmentTree(maxlen=capacity)

    def push(self, *experience):
        raise NotImplementedError

    def sample(self, batch_size, beta):
        indices = []
        for i in range(batch_size):
            segment = self._sum_tree.sum() / batch_size
            upper_bound = random.uniform(segment * i, segment * (i + 1))
            index = self._sum_tree.retrieve_index(upper_bound)
            indices.append(index)
        # Compute min P_i = min p_i ^ alpha / sum p_i ^ alpha
        min_probability = self._min_tree.min() / self._sum_tree.sum()
        # Compute max w_i = (N * min P_i) ^ (-beta)
        max_weight = (len(self.memory) * min_probability) ** (-bata)

        weights = []
        for index in indices:
            # Compute P_i = p_i ^ alpha / sum p_i ^ alpha
            probability = self._sum_tree[index] / self._sum_tree.sum()
            # Compute w_i = (N * P_i) ^ (-beta)
            weight = (len(self.memory) * probability) ** (-beta)
            # Normalized by max w_i
            weight /= max_weight
            weights.append(weight)

    def update(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            # Update max priority, max p_i
            self.max_priority = max(self.max_priority, priority)
            # Compute p_i ^ alpha
            priority_alpha = priority ** self.alpha
            # Update min segment tree
            self._min_tree[index] = priority_alpha
            # Update sum segment tree
            self._sum_tree[index] = priority_alpha


class TransitionReplayBuffer(ReplayBuffer):

    def push(self, *transition):
        self.memory.append(Transition(*transition))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def to_episode(self):
        return Episode(*zip(*self.memory))


class EpisodeReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, /, update='random'):
        super().__init__(capacity)
        if update not in ('random', 'sequential'):
            raise ValueError
        self.update = update

    def push(self, *episode):
        self.memory.append(Episode(*episode))

    def sample(self, batch_size):
        episodes = random.sample(self.memory, batch_size)
        if self.update == 'random':
            return Episode(*zip(*episodes))
        if self.update == 'sequential':
            return Episode(*zip(*episodes))
