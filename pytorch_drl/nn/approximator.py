#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class Approximator(torch.nn.Module):
    pass


class MLPApproximator(Approximator):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_sizes,
        out_activation=torch.nn.Identity(),
        hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        hidden_sizes = hidden_sizes['hidden_sizes']
        # Multilayer perceptron
        layers = []
        sizes = (in_dim,) + tuple(hidden_sizes) + (out_dim,)
        for i in range(len(sizes) - 2):
            layers += [
                torch.nn.Linear(sizes[i], sizes[i + 1], bias=True),
                hidden_activation,
            ]
        if out_dim is not None:
            layers += [
                torch.nn.Linear(sizes[-2], sizes[-1], bias=True),
                out_activation,
            ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, hc=None):
        return self.layers(x), hc


class RecurrentApproximator(Approximator):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_sizes,
        out_activation=torch.nn.Identity(),
        hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        pre_lstm_hidden_sizes = hidden_sizes['pre_lstm']
        lstm_hidden_sizes = hidden_sizes['lstm']
        post_lstm_hidden_sizes = hidden_sizes['post_lstm']
        # Pre-LSTM layers, perceptron
        self.pre_lstm_layers = MLPApproximator(
            in_dim,
            None,
            pre_lstm_hidden_sizes,
            hidden_activation=hidden_activation,
        )
        # LSTM layers
        lstm_layers = []
        lstm_sizes = (pre_lstm_hidden_sizes[-1],) + tuple(lstm_hidden_sizes)
        for i in range(len(lstm_sizes) - 1):
            lstm_layers += [
                torch.nn.LSTM(lstm_sizes[i], lstm_sizes[i + 1], batch_first=True),
            ]
        self.lstm_layers = torch.nn.Sequential(*lstm_layers)
        # Post-LSTM layers, perceptron
        self.post_lstm_layers = MLPApproximator(
            lstm_hidden_sizes[-1],
            out_dim,
            post_lstm_hidden_sizes,
            out_activation=out_activation,
            hidden_activation=hidden_activation,
        )

    def forward(self, x, hc=None):
        x = self.pre_lstm_layers(x)
        x, hc = self.lstm_layers(x, hc)
        x = x[:, -1, :]
        x = self.post_lstm_layers(x)
        return x, hc


class MemorizedMLPApproximator(Approximator):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_sizes,
        out_activation=torch.nn.Identity(),
        hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__()
        hidden_sizes = hidden_sizes['hidden_sizes']
        pre_lstm_hidden_sizes = hidden_sizes['pre_lstm']
        lstm_hidden_sizes = hidden_sizes['lstm']
        post_lstm_hidden_sizes = hidden_sizes['post_lstm']
        perceptron_hidden_sizes = hidden_sizes['perceptron']
        # Current input embedding, features
        self.layers = MLPApproximator(
            in_dim,
            None,
            hidden_sizes,
            hidden_activation=hidden_activation,
        )
        # Historical inputs embedding, memory
        self.memory_layers = RecurrentApproximator(
            in_dim,
            None,
            pre_lstm_hidden_sizes,
            lstm_hidden_sizes,
            post_lstm_hidden_sizes,
            hidden_activation=hidden_activation,
        )
        # Perceptron
        self.perceptron = MLPApproximator(
            hidden_sizes[-1] + post_lstm_hidden_sizes[-1],
            out_dim,
            perceptron_hidden_sizes,
            out_activation=out_activation,
            hidden_activation=hidden_activation,
        )

    def forward(self, x, history, hc=None):
        x = self.layers(x)
        memorized, hc = self.memory_layers(history, hc)
        # TODO: handle length variance within batch
        memorized = memorized[:, -1, :]
        x = torch.cat((x, memorized), dim=-1)
        x = self.perceptron(x)
        return x, hc
