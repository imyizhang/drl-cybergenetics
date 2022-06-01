#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import pathlib
import time

import torch


class Agent(abc.ABC):

    def __init__(self, device, actor, critic, buffer, actor_lr, critic_lr, batch_size, discount):
        self.device = device
        self.actor = actor.to(device) if actor is not None else None
        self.critic = critic.to(device) if critic is not None else None
        self.buffer = buffer if buffer is not None else None
        self.actor_optimizer = actor.configure_optimizer(actor_lr) if (actor is not None) and (actor_lr is not None) else None
        self.actor_criterion = actor.configure_criterion() if actor is not None else None
        self.critic_optimizer = critic.configure_optimizer(critic_lr) if (critic is not None) and (critic_lr is not None) else None
        self.critic_criterion = critic.configure_criterion() if critic is not None else None
        self.batch_size = batch_size if batch_size is not None else None
        self.gamma = discount if discount is not None else None

    def to(self, device):
        # TODO: handle device
        raise NotImplementedError

    def cache(self, *transition):
        self.buffer.push(*transition)

    def recall(self):
        batch = self.buffer.sample(self.batch_size)
        # Note: the last element in transition is used to discriminate trajectories
        return tuple(*[self._to(tensor, self.device) for tensor in self._as_tensors(batch[:-1])])

    @staticmethod
    def _to(tensor, device):
        if tensor is None:
            return None
        return tensor.to(device)

    @staticmethod
    def _as_tensors(batch):
        _lengths, lengths = None, None
        # Handle lengths for recurrent updates
        if not isinstance(batch[0][0], tuple):
            _lengths = [len(sequence) for sequence in batch[0]]
            lengths = torch.tensor(_lengths).view(len(_lengths), -1, 1)
        return [self._cat(field, _lengths) for field in batch] + [lengths]

    @staticmethod
    def _cat(field, lengths=None):
        if not isinstance(field[0], tuple):
            return torch.cat(field, dim=0)
        # Handle padding for recurrent updates
        sequences = [torch.cat(sequence, dim=0) for sequence in field]
        return torch.nn.utils.rnnpad_sequence(sequences, batch_first=True, padding_value=0.0)
        #return torch.nn.utils.pack_padded_sequence(paded, lengths, batch_first=True, enforce_sorted=False)

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError

    @staticmethod
    def _freeze(nn):
        for param in nn.parameters():
            param.requires_grad = False

    @staticmethod
    def _unfreeze(nn):
        for param in nn.parameters():
            param.requires_grad = True

    @staticmethod
    def _update(optimizer, loss, grad_clipping=False, grad_limit=1.0, nn=None):
        optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
            for param in nn.parameters():
                param.grad.data.clamp_(min=-grad_limit, max=grad_limit)
        optimizer.step()

    @staticmethod
    def _sync(nn_target, nn, soft_updating=False, tau=1e-3):
        # weights of target networks are updated by having them slowly track the learned networks with tau << 1
        if soft_updating:
            for param_target, param in zip(nn_target.parameters(), nn.parameters()):
                param_target.data.copy_(tau * param.data + (1.0 - tau) * param_target.data)
        # updated by directly copying the weights
        else:
            nn_target.load_state_dict(nn.state_dict())

    def save(self, dir):
        dir = pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        agent = self.__class__.__name__.lower()
        timestamp = str(int(time.time()))
        if self.actor is not None:
            filename = '.'.join([agent, 'actor', 'checkpoint', timestamp, 'pt'])
            torch.save(self.actor.state_dict(), dir.joinpath(filename))
        if self.critric is not None:
            filename = '.'.join([agent, 'critic', 'checkpoint', timestamp, 'pt'])
            torch.save(self.critric.state_dict(), dir.joinpath(filename))

    def load(self, path):
        path = pathlib.Path(path)
        if self.actor is not None:
            self.actor.load_state_dict(torch.load(path))
        if self.critric is not None:
            self.critric.load_state_dict(torch.load(path))
