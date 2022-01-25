#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class MLPApproximator(torch.nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        activation=torch.nn.ReLU(),
        out_activation=torch.nn.Identity(),
    ):
        super().__init__()
        dims = (in_dim,) + hidden_dim + (out_dim,)
        n_affine_maps = len(dims) - 1
        layers = []
        for i in range(n_affine_maps):
            nonlinearity = activation if i < n_affine_maps - 1 else out_activation
            layers += [
                torch.nn.Linear(dims[i], dims[i + 1], bias=True),
                nonlinearity,
             ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
