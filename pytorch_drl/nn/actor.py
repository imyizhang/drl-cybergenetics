#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

import torch

from .approximator import MLPApproximator


class BaseActor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #self.approximator = approximator

    @abc.abstractmethod
    def forward(self, state):
        #return self.approximator(state)
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, state, action_noise):
        # return action
        raise NotImplementedError

    # return optimizer
    def configure_optimizer(self, lr):
        return None

    # return criterion
    def configure_criterion(self):
        return None


class DummyActor(BaseActor):

    def __init__(self):
        super().__init__()

    def forward(self, state):
        raise RuntimeError


class ConstantActor(DummyActor):

    def __init__(self, action_dim, value=None):
        super().__init__()
        # continuous action space
        self.action = torch.empty(size=(1, action_dim)).uniform_(0, 1) if value is None else torch.tensor([[value]])

    def act(self, state):
        return self.action.to(device=state.device)


class RandomActor(DummyActor):

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def act(self, state):
        # continuous action space
        action = torch.empty(size=(1, self.action_dim)).uniform_(0, 1)
        return action.to(device=state.device)


class DiscreteConstantActor(DummyActor):

    def __init__(self, action_dim, value=None):
        super().__init__()
        # discrete action space
        self.action = torch.randint(
            low=0,
            high=action_dim,
            size=(1, 1),
        ) if value is None else torch.tensor([[value]])

    def act(self, state):
        return self.action.to(device=state.device)


class DiscreteRandomActor(DummyActor):

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def act(
        self,
        state,
        p=None,
    ):
        # discrete action space
        if p is None:
            action = torch.randint(
                low=0,
                high=self.action_dim,
                size=(1, 1),
            )
        else:
            action = torch.multinomial(p.view(-1), 1).view(1, 1)
        return action.to(device=state.device)


class Actor(BaseActor):

    def __init__(
        self,
        state_dim,
        action_dim,
        approximator_dims=(256, 256,),
        approximator_activation=torch.nn.Sigmoid(),
        approximator=MLPApproximator,
    ):
        super().__init__()
        self.approximator = approximator(
            state_dim,
            action_dim,
            approximator_dims,
            out_activation=approximator_activation,
        )

    def forward(self, state):
        return self.approximator(state)

    def act(
        self,
        state,
        action_noise,
        noise_clipping=False,
        c=0.5,
    ):
        with torch.no_grad():
            # action, continuous action space
            action = self(state)
            # noise
            noise = torch.randn_like(action) * action_noise
            # noise clipping
            if noise_clipping:
                noise = torch.clamp(noise, min=-c, max=c)
            # add noise
            action += noise
            # action clipping
            action = torch.clamp(action, min=0.0, max=1.0)
            return action

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)
