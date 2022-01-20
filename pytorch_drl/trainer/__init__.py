#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .trainer import Trainer
from .dummy import DummyTrainer
from .off_policy import OffPolicyTrainer

__all__ = (
    'Trainer',
    'DummyTrainer',
    'OffPolicyTrainer',
)
