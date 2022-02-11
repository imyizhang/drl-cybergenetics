#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


def replay(env, logger, episode=-1):
    return env.render(
        mode='dashboard',
        cache=logger.cache[episode],
    )

def heatmap(logger, episode=-1, file=None):
    Q = np.array(logger.episodic_Q[episode])
    s = np.array(logger.observations[episode]).reshape(-1)
    a = np.array(logger.actions[episode])
    df = pd.DataFrame(data=np.stack([s, a, Q]).T, columns=['s', 'a', 'Q'])
    pivotted = df.pivot('s', 'a', 'Q')
    sns.heatmap(pivotted)
    plot.show()
