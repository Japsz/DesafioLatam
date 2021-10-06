#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_csv('iris.csv')
X_mat = df.loc[:, ['x5_1', 'x3_5', 'x1_4', 'x0_2']]
y_vec = df['x0']

def calculate_phi(y_vec):
    """TODO: Docstring for calculate_phi.

    :y_vec: TODO
    :returns: TODO

    """
    m = len(y_vec)
    return np.where(x == 1, 1, 0) / m

