#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


def replay(env, logger, episode=-1):
    env.render(
        render_mode='dashboard',
        actions=logger.actions[episode],
        actions_taken=logger.actions_taken[episode],
        trajectory=logger.trajectories[episode],
        observations=logger.observations[episode],
        rewards=logger.rewards[episode],
        steps_done=logger.durations[episode],
    )
