#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# Parameters
TF_tot = 2000
k_on = 0.0016399
k_off = 0.34393
k_max = 13.588
K_d = 956.75
n = 4.203
k_basal = 0.02612
k_degR = 0.042116
k_trans = 1.4514
k_degP = 0.007


# ODEs
def yeast_odes(y, u):
    TF_on, mRNA, Protein = y
    I = u
    dTF_ondt = I * k_on * (TF_tot - TF_on) - k_off * TF_on
    dmRNAdt = k_basal + k_max * (TF_on ^ n) / (K_d ^ n + TF_on ^ n) - k_degR * mRNA
    dProteindt = k_trans * mRNA - k_degP * Protein
    return np.array([dTF_ondt, dmRNAdt, dProteindt])


# Initialization
yeast_init = np.array([
    0.0,
    k_basal / k_degR,
    (k_trans * k_basal) / (k_degP * k_degR),
])


# Environment configuration
yeast_config = {
    'physics': {
        'odes': yeast_odes,
        'init': yeast_init,
        'dtype': np.float32,
    },
    'task': {

    },
    'environment': {

    }
}


# Parameters
d_r = 0.0956
d_p = 0.0214
k_m = 0.0116
b_r = 0.0965


# ODEs
def ecoli_odes(y, u):
    a = np.array([[1],
                  [u]])
    A_c = np.array([[-d_r, 0.0, 0.0],
                    [d_p + k_m, -d_p - k_m, 0.0],
                    [0.0, d_p, -d_p]])
    B_c = np.array([[d_r, b_r],
                    [0.0, 0.0],
                    [0.0, 0.0]])
    dydt = A_c @ y + B_c @ a
    return dydt


# Initialization
ecoli_init = np.array([1.0, 1.0, 1.0])


# Environment configuration
ecoli_config = {
    'physics': {
        'odes': ecoli_odes,
        'init': ecoli_init,
        'dtype': np.float32,
    },
    'task': {

    },
    'environment': {

    }
}
