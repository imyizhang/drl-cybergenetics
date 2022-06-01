#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class Approximator(torch.nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        out_activation,
        hidden_activation,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.out_activation = out_activation
        self.hidden_activation = hidden_activation
        # Define nn layers: `self.layers = ...`

    def forward(self, x):
        raise NotImplementedError


class FeedforwardApproximator(Approximator):
    """Simple feedforward neural network based on multilayer perceptron (MLP)."""

    def __init__(
        self,
        input_size,
        output_size=None,
        hidden_sizes={
            'mlp': (256,),
        },
        out_activation=torch.nn.Identity(),
        hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__(
            input_size,
            output_size,
            hidden_sizes,
            out_activation,
            hidden_activation,
        )
        mlp_hidden_sizes = self.hidden_sizes['mlp']
        # MLP layers
        layers = []
        sizes = (self.input_size,) + tuple(mlp_hidden_sizes) + (self.output_size,)
        for i in range(len(sizes) - 2):
            layers += [
                torch.nn.Linear(sizes[i], sizes[i + 1], bias=True),
                self.hidden_activation,
            ]
        if self.output_size is not None:
            layers += [
                torch.nn.Linear(sizes[-2], sizes[-1], bias=True),
                self.out_activation,
            ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, hc=None, lengths=None):
        return self.layers(x), hc


class RecurrentApproximator(Approximator):
    """Simple recurrent neural network based on long short term memory (LSTM)."""

    def __init__(
        self,
        input_size,
        output_size=None,
        hidden_sizes={
            'pre_lstm': (256,),
            'lstm': (256,),
            'post_lstm': (256,),
        },
        out_activation=torch.nn.Identity(),
        hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__(
            input_size,
            output_size,
            hidden_sizes,
            out_activation,
            hidden_activation,
        )
        pre_lstm_hidden_sizes = self.hidden_sizes['pre_lstm']
        lstm_hidden_sizes = self.hidden_sizes['lstm']
        post_lstm_hidden_sizes = self.hidden_sizes['post_lstm']
        # Pre-LSTM MLP layers
        self.pre_lstm_layers = FeedforwardApproximator(
            self.input_size,
            hidden_sizes={
                'mlp': pre_lstm_hidden_sizes,
            },
            hidden_activation=self.hidden_activation,
        )
        # LSTM layers
        lstm_layers = []
        lstm_sizes = (pre_lstm_hidden_sizes[-1],) + tuple(lstm_hidden_sizes)
        for i in range(len(lstm_sizes) - 1):
            lstm_layers += [
                torch.nn.LSTM(lstm_sizes[i], lstm_sizes[i + 1], batch_first=True),
            ]
        self.lstm_layers = torch.nn.Sequential(*lstm_layers)
        # Post-LSTM MLP layers
        self.post_lstm_layers = FeedforwardApproximator(
            lstm_hidden_sizes[-1],
            self.output_size,
            hidden_sizes={
                'mlp': post_lstm_hidden_sizes,
            },
            out_activation=self.out_activation,
            hidden_activation=self.hidden_activation,
        )

    def forward(self, x, hc=None, lengths=None):
        x, _ = self.pre_lstm_layers(x)
        x, hc = self.lstm_layers(x, hc)
        x, _ = self.post_lstm_layers(x)
        # x, _ = torch.nn.utils.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        # Equivalent to `x = x[:, -1, :]` if sequences within batch have the same length
        if lengths is not None:
            # Make sure `lengths.shape == (batch_size, 1)`
            x = x.gather(dim=1, index=lengths - 1).squeeze(dim=1)
        return x, hc


class MemorizedMLP(Approximator):

    def __init__(
        self,
        input_size,
        output_size=None,
        hidden_sizes={
            'mlp1': (256,),
            'pre_lstm': (256,),
            'lstm': (256,),
            'post_lstm': (256,),
            'mlp2': (256,),
        },
        out_activation=torch.nn.Identity(),
        hidden_activation=torch.nn.ReLU(),
    ):
        super().__init__(
            input_size,
            output_size,
            hidden_sizes,
            out_activation,
            hidden_activation,
        )
        mlp1_hidden_sizes = self.hidden_sizes['mlp1']
        pre_lstm_hidden_sizes = self.hidden_sizes['pre_lstm']
        lstm_hidden_sizes = self.hidden_sizes['lstm']
        post_lstm_hidden_sizes = self.hidden_sizes['post_lstm']
        mlp2_hidden_sizes = self.hidden_sizes['mlp2']
        # Current observation embedding, observing features
        self.observer_layers = FeedforwardApproximator(
            self.input_size,
            hidden_sizes={
                'mlp': mlp1_hidden_sizes,
            },
            hidden_activation=self.hidden_activation,
        )
        # Historical observations embedding, memorized features
        self.memory_layers = RecurrentApproximator(
            self.input_size,
            hidden_sizes={
                'pre_lstm': pre_lstm_hidden_sizes,
                'lstm': lstm_hidden_sizes,
                'post_lstm': post_lstm_hidden_sizes,
            },
            hidden_activation=self.hidden_activation,
        )
        # MLP layers
        self.layers = FeedforwardApproximator(
            mlp1_hidden_sizes[-1] + post_lstm_hidden_sizes[-1],
            self.output_size,
            hidden_sizes={
                'mlp': mlp2_hidden_sizes,
            },
            out_activation=self.out_activation,
            hidden_activation=self.hidden_activation,
        )

    def forward(self, x, hc=None, lengths=None):
        observing, _ = self.observer_layers(x[:, -1, :])
        # Make sure `lengths is not None`
        memorized, hc = self.memory_layers(x[:, :-1, :], hc, lengths)
        x = torch.cat((observing, memorized), dim=-1)
        x, _ = self.layers(x)
        return x, hc
