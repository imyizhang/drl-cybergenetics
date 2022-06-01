#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

import torch

from .approximator import FeedforwardApproximator


class Actor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Define approximator: `self.approximator = ...`

    @abc.abstractmethod
    def forward(self, state):
        # return self.approximator(state)
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, state):
        # return action
        raise NotImplementedError

    def configure_optimizer(self, lr):
        return None

    def configure_criterion(self):
        return None


class DummyActor(Actor):

    def forward(self, state):
        raise RuntimeError


class ConstantActor(DummyActor):
    """Constant pi(s) in continuous action space."""

    def __init__(self, action_dim, action_limit=1.0, value=None):
        super().__init__()
        if value is None:
            self.action = torch.empty(size=(1, action_dim)).uniform_(-action_limit, action_limit)
        else:
            self.action = torch.tensor([[value]])

    def act(self, state):
        return self.action.to(device=state.device, dtype=state.dtype)


class RandomActor(DummyActor):
    """Uniform pi(s) distribution in continuous action space."""

    def __init__(self, action_dim, action_limit=1.0):
        super().__init__()
        self.action_dim = action_dim
        self.action_limit = action_limit

    def act(self, state):
        action = torch.empty(size=(1, self.action_dim)).uniform_(-self.action_limit, self.action_limit)
        return action.to(device=state.device, dtype=state.dtype)


class DiscreteConstantActor(DummyActor):
    """Constant pi(s) in discrete action space."""

    def __init__(self, action_dim, value=None):
        super().__init__()
        if value is None:
            self.action = torch.randint(low=0, high=action_dim, size=(1, 1))
        else:
            self.action = torch.tensor([[value]])

    def act(self, state):
        return self.action.to(device=state.device, dtype=state.dtype)


class DiscreteRandomActor(DummyActor):
    """Uniform pi(s) distribution in discrete action space."""

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def act(self, state):
        action = torch.randint(low=0, high=self.action_dim, size=(1, 1))
        return action.to(device=state.device, dtype=state.dtype)


class DeterministicActor(Actor):
    """Estimate deterministic pi(s) in continuous action space."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Tanh(),
        approximator_hidden_activation=torch.nn.ReLU(),
        action_limit=1.0,
    ):
        super().__init__()
        self.approximator = approximator(
            state_dim,
            action_dim,
            approximator_hidden_sizes,
            out_activation=approximator_out_activation,
            hidden_activation=approximator_hidden_activation,
        )
        self.action_dim = action_dim
        self.action_limit = action_limit

    def forward(self, state, **kwargs):
        pi, hc = self.approximator(state, **kwargs)
        return self.action_limit * pi, hc

    def act(self, state, action_noise, noise_clipping=False, noise_limit=0.5, **kwargs):
        with torch.no_grad():
            action, _ = self(state, **kwargs)
        noise = torch.randn_like(action) * action_noise
        if noise_clipping:
            noise = torch.clamp(noise, min=-noise_limit, max=noise_limit)
        action += noise
        action = torch.clamp(action, min=-self.action_limit, max=self.action_limit)
        return action

    def explore(self, state):
        action = torch.empty(size=(1, self.action_dim)).uniform_(-self.action_limit, self.action_limit)
        return action.to(device=state.device, dtype=state.dtype)

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class StochasticActor(Actor):
    """Estimate stochastic pi(s) distribution in continuous action space."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Tanh(),
        approximator_hidden_activation=torch.nn.ReLU(),
        action_limit=1.0,
    ):
        super().__init__()
        self.approximator = approximator(
            state_dim,
            action_dim,
            approximator_hidden_sizes,
            out_activation=approximator_out_activation,
            hidden_activation=approximator_hidden_activation,
        )
        log_sigma = -0.5 * torch.ones(action_dim)
        self.log_sigma = torch.nn.Parameter(log_sigma)

    def forward(self, state, **kwargs):
        mu = self.approximator(state, **kwargs)
        sigma = torch.exp(self.log_sigma)
        return torch.distributions.Normal(mu, sigma)

    def log_prob(self, pi_distribution, action):
        return pi_distribution.log_prob(action).sum(axis=-1, keepdim=True)

    def entropy(self, pi_distribution, action):
        return pi_distribution.entropy(action).sum(axis=-1, keepdim=True)

    def step(self, state, **kwargs):
        with torch.no_grad():
            pi_distribution = self(state, **kwargs)
            action = pi_distribution.sample()
            log_prob = self.log_prob(pi_distribution, action)
        return action, log_prob

    def act(self, state, **kwargs):
        with torch.no_grad():
            pi_distribution = self(state, **kwargs)
            action = pi_distribution.sample()
        return action

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class DiscreteStochasticActor(Actor):
    """Estimate stochastic pi(s) distribution in discrete action space."""

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'mlp': (256, 256,)
        },
        approximator_out_activation=torch.nn.Tanh(),
        approximator_hidden_activation=torch.nn.ReLU(),
        action_limit=1.0,
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
        logits = self.approximator(state, **kwargs)
        return torch.distributions.Categorical(logits=logits)

    def log_prob(self, pi_distribution, action):
        return pi_distribution.log_prob(action).view(-1, 1)

    def entropy(self, pi_distribution):
        return pi_distribution.entropy().view(-1, 1)

    def step(self, state, **kwargs):
        with torch.no_grad():
            pi_distribution = self(state, **kwargs)
            action = pi_distribution.sample()
            log_prob = self.log_prob(pi_distribution, action)
        return action, log_prob

    def act(self, state, **kwargs):
        with torch.no_grad():
            pi_distribution = self(state, **kwargs)
            action = pi_distribution.sample()
        return action

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class SquashedActor(Actor):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator=FeedforwardApproximator,
        approximator_hidden_sizes={
            'embedding': {
                'mlp': (256, 256,)
            },
            'mu': {
                'mlp': (256, 256,)
            },
            'log_sigma': {
                'mlp': (256, 256,)
            },
        },
        approximator_out_activation=torch.nn.Tanh(),
        approximator_hidden_activation=torch.nn.ReLU(),
        action_limit=1.0,
    ):
        super().__init__()
        embedding_hidden_sizes = approximator_hidden_sizes['embedding']
        self.embedding_approximator = approximator(
            state_dim,
            embedding_hidden_sizes,
            hidden_activation=approximator_hidden_activation,
        )
        self.mu_approximator = FeedforwardApproximator(
            embedding_hidden_sizes[list(embedding_hidden_sizes.keys())[-1]][-1],
            action_dim,
            approximator_hidden_sizes['mu'],
            out_activation=approximator_out_activation,
            hidden_activation=approximator_hidden_activation,
        )
        self.log_sigma_approximator = FeedforwardApproximator(
            embedding_hidden_sizes[list(embedding_hidden_sizes.keys())[-1]][-1],
            action_dim,
            approximator_hidden_sizes['log_sigma'],
            out_activation=approximator_out_activation,
            hidden_activation=approximator_hidden_activation,
        )
        self.action_dim = action_dim
        self.action_limit = action_limit

    def forward(self, state, stochastic=True, with_log_prob=True, **kwargs):
        log_sigma_min = -20
        log_sigma_max = 2
        embedding, hc = self.embedding_approximator(state, **kwargs)
        mu = self.mu_approximator(embedding)
        log_sigma = self.log_sigma_approximator(embedding)
        log_sigma = torch.clamp(log_sigma, log_sigma_min, log_sigma_max)
        sigma = torch.exp(log_sigma)
        pi_distribution = torch.distributions.Normal(mu, sigma)
        action = pi_distribution.rsample() if stochastic else mu
        if with_log_prob:
            log_prob = pi_distribution.log_prob(action).sum(axis=-1)
            log_prob -= (2 * torch.log(torch.tensor([2])) - action - torch.nn.functional.softplus(-2 * action)).sum(axis=1)
        else:
            log_prob = None
        action = torch.tanh(action)
        action = self.act_limit * action
        return action, log_prob, hc

    def act(self, state, stochastic, **kwargs):
        with torch.no_grad():
            action, _, _ = self(state, stochastic, with_log_prob=False, **kwargs)
        return action

    def explore(self, state):
        action = torch.empty(size=(1, self.action_dim)).uniform_(-self.action_limit, self.action_limit)
        return action.to(device=state.device, dtype=state.dtype)

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)
