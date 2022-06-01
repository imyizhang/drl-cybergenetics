#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

import torch

from .approximator import FeedforwardApproximator


class Critic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Define approximator: `self.approximator = ...`

    @abc.abstractmethod
    def forward(self, state, action):
        # return self.approximator(state, action)
        raise NotImplementedError

    def configure_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def configure_criterion(self):
        return torch.nn.SmoothL1Loss()


class VCritic(Critic):
    """Estimate V(s) in continuous action space based on universal
    approximation theorem."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Identity(),
        approximator_hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.approximator = approximator(
            state_dim,
            action_dim,
            approximator_hidden_sizes,
            out_activation=approximator_out_activation,
            hidden_activation=approximator_hidden_activation,
        )

    def forward(self, state, **kwargs):
        return self.approximator(state, **kwargs)


class QCritic(Critic):
    """Estimate Q(s, a) in continuous action space based on universal
    approximation theorem."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Identity(),
        approximator_hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.approximator = approximator(
            state_dim + action_dim,
            1,
            approximator_hidden_sizes,
            out_activation=approximator_out_activation,
            hidden_activation=approximator_hidden_activation,
        )

    def forward(self, state, action, **kwargs):
        return self.approximator(torch.cat((state, action), dim=-1), **kwargs)


class TwinQCritic(Critic):
    """Double estimate Q(s, a) in continuous action space based on universal
    approximation theorem."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Identity(),
        approximator_hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.critic1 = QCritic(
            state_dim,
            action_dim,
            approximator,
            approximator_hidden_sizes,
            approximator_out_activation,
            approximator_hidden_activation,
        )
        self.critic2 = QCritic(
            state_dim,
            action_dim,
            approximator,
            approximator_hidden_sizes,
            approximator_out_activation,
            approximator_hidden_activation,
        )

    def forward(self, state, action, **kwargs):
        return self.critic1(self, state, action, **kwargs),

    def get_twin(self, state, action, **kwargs):
        # Critic returns tuple by default
        return (
            self.critic1(self, state, action, **kwargs)[0],
            self.critic2(self, state, action, **kwargs)[0],
        )


class DiscreteQCritic(Critic):
    """Estimate Q(s, a) distribution in discrete action space based on universal
    approximation theorem."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Identity(),
        approximator_hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.approximator = approximator(
            state_dim,
            action_dim,
            approximator_hidden_sizes,
            out_activation=approximator_out_activation,
            hidden_activation=approximator_hidden_activation,
        )

    def forward(self, state, **kwargs):
        return self.approximator(state, **kwargs)


class DiscreteTwinQCritic(Critic):
    """Double estimate Q(s, a) distribution in discrete action space based on
    universal approximation theorem."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Identity(),
        approximator_hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.critic1 = DiscreteQCritic(
            state_dim,
            action_dim,
            approximator,
            approximator_hidden_sizes,
            approximator_out_activation,
            approximator_hidden_activation,
        )
        self.critic2 = DiscreteQCritic(
            state_dim,
            action_dim,
            approximator,
            approximator_hidden_sizes,
            approximator_out_activation,
            approximator_hidden_activation,
        )

    def forward(self, state, **kwargs):
        return self.critic1(state, **kwargs)

    def get_twin(self, state, **kwargs):
        # Critic returns tuple by default
        return (
            self.critic1(state, **kwargs)[0],
            self.critic2(state, **kwargs)[0],
        )


class DiscreteDuelingQCritic(Critic):
    """Estimate Q(s, a) distribution in discrete action space using dueling
    architectures."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'embedding': {
                'mlp': (256, 256,)
            },
            'advantage': {
                'mlp': (256, 256,)
            },
            'V': {
                'mlp': (256, 256,)
            },
        },
        approximator_out_activation=torch.nn.Identity(),
        approximator_hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        embedding_hidden_sizes = approximator_hidden_sizes['embedding']
        self.embedding_approximator = approximator(
            state_dim,
            embedding_hidden_sizes,
            hidden_activation=approximator_hidden_activation,
        )
        self.advantage_approximator = FeedforwardApproximator(
            embedding_hidden_sizes[list(embedding_hidden_sizes.keys())[-1]][-1],
            action_dim,
            approximator_hidden_sizes['advantage'],
            out_activation=approximator_out_activation,
            hidden_activation=approximator_hidden_activation,
        )
        self.V_approximator = FeedforwardApproximator(
            embedding_hidden_sizes[list(embedding_hidden_sizes.keys())[-1]][-1],
            1,
            approximator_hidden_sizes['V'],
            hidden_activation=approximator_hidden_activation,
        )

    def forward(self, state, **kwargs):
        embedding, hc = self.embedding_approximator(state, **kwargs)
        advantage = self.advantage_approximator(embedding)
        V = self.V_approximator(embedding)
        return V + advantage - advantage.mean(), hc


class DiscreteTwinDuelingQCritic(Critic):
    """Double estimate Q(s, a) distribution in discrete action space using
    dueling architectures."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Identity(),
        approximator_hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.critic1 = DiscreteDuelingQCritic(
            state_dim,
            action_dim,
            approximator,
            approximator_hidden_sizes,
            approximator_out_activation,
            approximator_hidden_activation,
        )
        self.critic2 = DiscreteDuelingQCritic(
            state_dim,
            action_dim,
            approximator,
            approximator_hidden_sizes,
            approximator_out_activation,
            approximator_hidden_activation,
        )

    def forward(self, state, **kwargs):
        return self.critic1(state, **kwargs)

    def get_twin(self, state, **kwargs):
        # Critic returns tuple by default
        return (
            self.critic1(state, **kwargs)[0],
            self.critic2(state, **kwargs)[0],
        )
