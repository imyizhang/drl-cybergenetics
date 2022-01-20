#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .agent import Agent
from .dummy import DummyAgent
# off-policy
from .dqn import DQNAgent, DDQNAgent
from .ddpg import DDPGAgent, TD3Agent

__all__ = (
    'Agent',
    'DummyAgent',
    'DQNAgent',
    'DDQNAgent',
    'DDPG',
    'TD3Agent',
)
