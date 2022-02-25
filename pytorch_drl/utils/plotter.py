#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


def replay(env, logger, episode=-1):
    return env.render(mode='dashboard', cache=logger.episodic_cache[episode])

def plot(logger):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(6, 5),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    #fig.tight_layout()
    axs[0].plot([sum(r) for r in logger.episodic_rewards], marker='.', color='tab:orange')
    axs[0].set_ylabel('reward')
    axs[0].grid(True)
    axs[1].plot([sum(t) / len(t) * 100 for t in logger.episodic_aggregator_in_tolerance], marker='.', color='tab:red')
    axs[1].set_ylim([0, 100])
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('% states in tolerance zone')
    axs[1].grid(True)
    return fig

def heatmap(logger, episode=-1):
    Q = np.array(logger.episodic_Q[episode]).reshape(-1)
    s = np.array(logger.episodic_states[episode]).reshape(-1)
    a = np.array(logger.episodic_actions[episode]).reshape(-1)
    df = pd.DataFrame(data=np.stack([s, a, Q]).T, columns=['s', 'a', 'Q'])
    pivotted = df.pivot('s', 'a', 'Q')
    heatmap = sns.heatmap(pivotted)
    return heatmap.get_figure()
