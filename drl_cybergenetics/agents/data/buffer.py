#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import collections
import random
import itertools

from .segment_tree import SumSegmentTree, MinSegmentTree


def _randint(low, high):
    return random.randrange(low, high)


def _sample(population, k):
    if k > len(population):
        return random.choices(population, k=k)
    return random.sample(population, k=k)


def _slice(l, indices):
    return [l[i] for i in indices]


def _index(l, value):
    if value not in l:
        return None
    return len(l) - 1 - l[::-1].index(value)


def _value(l):
    u = list(set(l))
    if len(u) < 2:
        return None
    return u[-2]


class Buffer(abc.ABC):

    def __init__(self, capacity, push_fields, pop_fields=None):
        self.data = collections.deque([], maxlen=capacity)
        self.push_fields = push_fields
        self.pop_fields = pop_fields if pop_fields is not None else push_fields
        self.transition = collections.namedtuple('Transition', self.push_fields)
        self.trajectory = collections.namedtuple('Trajectory', self.push_fields)
        self.batch = collections.namedtuple('Batch', self.pop_fields)

    def __len__(self):
        return len(self.data)

    @property
    def _trajectories(self):
        # Note: the last element in transition is used to discriminate trajectories
        return [list(g) for _, g in itertools.groupby(self.data, key=lambda transition: transition[-1])]

    @property
    def trajectories(self):
        # Note: the last element in transition is used to discriminate trajectories
        return [self.trajectory(*zip(*g)) for _, g in itertools.groupby(self.data, key=lambda transition: transition[-1])]

    def push(self, *transition):
        self.data.append(self.transition(*transition))

    def clear(self):
        self.data.clear()

    def pop_all(self):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def seed(seed=None):
        random.seed(seed)


class RolloutBuffer(Buffer):

    @trajectories.setter
    def trajectories(self, value):
        self.pop_fields = ('state', 'action', 'reward_to_go', 'advantage', 'log_prob')
        self.trajectory = collections.namedtuple('Trajectory', self.pop_fields)
        self.batch = collections.namedtuple('Batch', self.pop_fields)
        self.trajectories = [self.trajectory(*zip(*t)) for t in value]

    def pop_all(self):
        # Note: pop fields are the same as push fields, normally including
        # 'state', 'action', 'reward', 'V', 'log_prob', 'done', 'trajectory_start'
        batch = self.batch(*zip(*self.data))
        self.clear()
        return batch

    def sample(self, batch_size, shuffle=True, only_indices=False):
        size = len(self.data)
        indices = list(range(size))
        if shuffle:
            random.shuffle(indices)
        if only_indices:
            return [indices[i:i + batch_size] for i in range(0, size, batch_size)]
        # Note: pop fields are the same as push fields, normally including
        # 'state', 'action', 'reward', 'V', 'log_prob', 'done', 'trajectory_start'
        return [self.batch(*zip(*_slice(self.data, indices[i:i + batch_size]))) for i in range(0, size, batch_size)]


class RecurrentRolloutBuffer(Buffer):

    def pop_all(self, max_history_steps=None):
        # Whole trajectories are packaged
        if max_history_steps is None:
            # Note: pop fields are the same as push fields, normally including
            # 'state', 'action', 'reward', 'V', 'log_prob', 'done', 'trajectory_start'
            # Note: padding, lengths to be handled for recurrent updates
            batch = self.batch(*zip(*self.trajectories))
        # Sub-trajectories are packaged
        else:
            trajectories = [self._split(t, max_history_steps + 1) for t in self._trajectories]
            # Note: pop fields are the same as push fields, normally including
            # 'state', 'action', 'reward', 'V', 'log_prob', 'done', 'trajectory_start'
            batch = self.batch(*zip(*itertools.chain(*trajectories)))
        self.clear()
        return batch

    def sample(self, batch_size, max_history_steps=None, shuffle=True, only_indices=False):
        # Whole trajectories are packaged
        if max_history_steps is None:
            trajectories = self.trajectories
        # Sub-trajectories are packaged
        else:
            trajectories = [self._split(t, max_history_steps + 1) for t in self._trajectories]
            # Flatten nested list
            trajectories = list(itertools.chain(*trajectories))
        size = len(trajectories)
        indices = list(range(size))
        if shuffle:
            random.shuffle(indices)
        if only_indices:
            return [indices[i:i + batch_size] for i in range(0, size, batch_size)]
        # Note: pop fields are the same as push fields, normally including
        # 'state', 'action', 'reward', 'V', 'log_prob', 'done', 'trajectory_start'
        return [self.batch(*zip(*_slice(trajectories, indices[i:i + batch_size]))) for i in range(0, size, batch_size)]

    def _split(self, trajectory, max_subsize, bootstrap):
        size = len(trajectory)
        # TODO: handle trajectory only containing single transition (rare occurrence)
        if size == 1:
            raise ValueError
        if max_subsize > size:
            # Note: padding, lengths to be handled for recurrent updates
            return self.trajectory(*zip(*trajectory))
        return [self.trajectory(*zip(*trajectory[i:i + max_subsize])) for i in range(0, size - max_subsize + 1)]


class ReplayBuffer(Buffer):

    def sample(self, batch_size):
        transitions = _sample(self.data, batch_size)
        # Note: pop fields are the same as push fields, normally including
        # 'state', 'action', 'reward', 'next_state', 'done', 'trajectory_start'
        return self.batch(*zip(*transitions))


class RecurrentReplayBuffer(Buffer):
    """Replay Buffer for Recurrent Updates.

    References:
        [1] 'Deep Recurrent Q-Learning for Partially Observable MDPs' (2015). arxiv.org/abs/1507.06527
    """

    def sample(self, batch_size, max_history_steps=None, sample_trajectory_first=True):
        # Whole trajectories are packaged for bootstrapped sequential updates
        if max_history_steps is None:
            # Note: padding, lengths to be handled for recurrent updates
            trajectories = _sample(self.trajectories, batch_size)
        # Random sub-trajectories are packaged for bootstrapped random updates:
        # either in a way that trajectories are sampled first
        elif sample_trajectory_first:
            trajectories = [self._choice(t, max_history_steps + 1) for t in _sample(self._trajectories, batch_size)]
        # or not
        else:
            trajectories = [self._choice_via_index(i, max_history_steps + 1) for i in _sample(range(len(self.data)), batch_size)]
        # Note: pop fields are the same as push fields, normally including
        # 'state', 'action', 'reward', 'next_state', 'done', 'trajectory_start'
        return self.batch(*zip(*trajectories))

    def _choice(self, trajectory, max_subsize):
        size = len(trajectory)
        # TODO: handle trajectory only containing single transition (rare occurrence)
        if size == 1:
            raise ValueError
        if max_subsize < size:
            index = _randint(max_subsize, size)
            return self.trajectory(*zip(*trajectory[index - max_subsize:index]))
        # Note: padding, lengths to be handled for recurrent updates
        return self.trajectory(*zip(*trajectory))

    def _choice_via_index(self, index, max_size):
        start_index = index - max_size + 1
        trajectory = self.trajectory(*zip(*list(self.data)[start_index:index + 1]))
        # Find index where the last trajectory is terminal
        # Note: the last element in transition is used to discriminate trajectories
        terminal_index = _index(trajectory[-1], _value(trajectory[-1]))
        if terminal_index is not None:
            start_index += terminal_index + 1
            # TODO: handle trajectory only containing single transition (rare occurrence)
            if index - start_index == 0:
                raise ValueError
            # Note: padding, lengths to be handled for recurrent updates
            trajectory = self.trajectory(*zip(*list(self.data)[start_index:index + 1]))
        return trajectory


class PrioritizedReplayBuffer(Buffer):
    """Prioritized Replay Buffer.

    References:
        [1] 'Prioritized Experience Replay' (2015). arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity, alpha, push_fields, pop_fields=None):
        super().__init__(self, capacity, push_fields, pop_fields)
        self.max_priority = 1.0
        self.alpha = alpha
        self._sum_tree = SumSegmentTree(maxlen=capacity)
        self._min_tree = MinSegmentTree(maxlen=capacity)

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
        max_weight = (len(self.data) * min_probability) ** (-bata)

        weights = []
        for index in indices:
            # Compute P_i = p_i ^ alpha / sum p_i ^ alpha
            probability = self._sum_tree[index] / self._sum_tree.sum()
            # Compute w_i = (N * P_i) ^ (-beta)
            weight = (len(self.data) * probability) ** (-beta)
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
