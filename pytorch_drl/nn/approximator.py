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
        # Define approximator
        #self.layers =

    def forward(self, x, history=None, hc=None):
        raise NotImplementedError


class MLPApproximator(Approximator):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes = {
            'hidden_sizes': (256, 256,),
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
        hidden_sizes = self.hidden_sizes['hidden_sizes']
        # Multilayer perceptron
        layers = []
        sizes = (self.input_size,) + tuple(hidden_sizes) + (self.output_size,)
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

    def forward(self, x, history=None, hc=None):
        return self.layers(x), hc


class RecurrentApproximator(Approximator):

    def __init__(
        self,
        input_size,
        output_size,
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
        # Pre-LSTM layers, perceptron
        self.pre_lstm_layers = MLPApproximator(
            self.input_size,
            None,
            {
                'hidden_sizes': pre_lstm_hidden_sizes,
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
        # Post-LSTM layers, perceptron
        self.post_lstm_layers = MLPApproximator(
            lstm_hidden_sizes[-1],
            self.output_size,
            {
                'hidden_sizes': post_lstm_hidden_sizes,
            },
            out_activation=self.out_activation,
            hidden_activation=self.hidden_activation,
        )

    def forward(self, x, history=None, hc=None):
        x = self.pre_lstm_layers(x)
        x, hc = self.lstm_layers(x, hc)
        x = x[:, -1, :]
        x = self.post_lstm_layers(x)
        return x, hc


class MemorizedMLPApproximator(Approximator):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes={
            'hidden_sizes': (256, 256,),
            'pre_lstm': (256,),
            'lstm': (256,),
            'post_lstm': (256,),
            'perceptron': (256,),
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
        hidden_sizes = self.hidden_sizes['hidden_sizes']
        pre_lstm_hidden_sizes = self.hidden_sizes['pre_lstm']
        lstm_hidden_sizes = self.hidden_sizes['lstm']
        post_lstm_hidden_sizes = self.hidden_sizes['post_lstm']
        perceptron_hidden_sizes = self.hidden_sizes['perceptron']
        # Current input embedding, features
        self.layers = MLPApproximator(
            self.input_size,
            None,
            {
                'hidden_sizes': hidden_sizes,
            },
            hidden_activation=self.hidden_activation,
        )
        # Historical inputs embedding, memorized features
        self.memory_layers = RecurrentApproximator(
            self.input_size,
            None,
            {
                'pre_lstm': pre_lstm_hidden_sizes,
                'lstm': lstm_hidden_sizes,
                'post_lstm': post_lstm_hidden_sizes,
            },
            hidden_activation=self.hidden_activation,
        )
        # Perceptron
        self.perceptron = MLPApproximator(
            hidden_sizes[-1] + post_lstm_hidden_sizes[-1],
            self.output_size,
            {
                'hidden_sizes': perceptron_hidden_sizes,
            },
            out_activation=self.out_activation,
            hidden_activation=self.hidden_activation,
        )

    def forward(self, x, history, hc=None):
        x = self.layers(x)
        memorized, hc = self.memory_layers(history, hc)
        # TODO: handle length variance within batch
        memorized = memorized[:, -1, :]
        x = torch.cat((x, memorized), dim=-1)
        x = self.perceptron(x)
        return x, hc
