#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec12_graphs.py
Author: Ignacio Soto Zamorano / Ignacio Loayza Campos
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com / ignacio1505[at]gmail[dot]com
Github: https://github.com/ignaciosotoz / https://github.com/tattoedeer
Description: Ancilliary files for Neural Network Design lecture - ADL
"""

import matplotlib.pyplot as plt

def plot_cross_entropy(history, ax=None):
    """TODO: Docstring for plot_cross_entropy.

    :history: TODO
    :returns: TODO

    """
    ax = ax or plt.gca()
    extract_val_loss = history.history['val_loss']
    extract_loss = history.history['loss']
    ax.plot(range(1, len(extract_val_loss) + 1),
            extract_val_loss, label = 'CV Error',
            color='tomato', lw=2)
    ax.plot(range(1, len(extract_loss) + 1),
             extract_loss, label = 'Training Error',
             color='dodgerblue', lw=2)
    return ax
