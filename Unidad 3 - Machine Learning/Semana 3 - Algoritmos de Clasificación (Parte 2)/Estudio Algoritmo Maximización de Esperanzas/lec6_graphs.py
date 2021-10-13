#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec6_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary files for Expectation Maximization Algorithm - ADL
"""

#######################################################################
#                        work environment prep                        #
#######################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

#######################################################################
#                               Helpers                               #
#######################################################################

fetch_lims = lambda x: [np.floor(np.min(x)), np.ceil(np.max(x))]

def handle_df(list_of_list, covar_type):
    tmp_df = pd.DataFrame(list_of_list).T
    tmp_df.columns = covar_type
    tmp_df['n_components'] = tmp_df.index
    return tmp_df

def get_joint_xy(xlim, ylim):
    """TODO: Docstring for get_joint_xy.

    :xlim: TODO
    :ylim: TODO
    :returns: TODO

    """
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(xlim[0], xlim[1]),
        np.linspace(ylim[0], ylim[1])
    )

    joint_xy = np.vstack([
        x_mesh.ravel(),
        y_mesh.ravel()
    ]).T

    return x_mesh, y_mesh, joint_xy

def plot_centroids(centers, weights):
    """TODO: Docstring for plot_centroids.

    :centers: TODO
    :weights: TODO
    :returns: TODO

    """
    tmp_centroids = centers[weights > weights.max() / 10]
    plt.scatter(tmp_centroids[:, 0], tmp_centroids[:, 1], edgecolor='slategrey', facecolor='white', s=150)
    plt.scatter(tmp_centroids[:, 0], tmp_centroids[:, 1], color='tomato', marker='o', s=50)

#######################################################################
#                    Lecture and diagnosis graphs                     #
#######################################################################


def plot_gaussian_ellipses(model, X):
    """TODO: Docstring for plot_gaussian_ellipses.

    :model: TODO
    :X: TODO
    :returns: TODO

    """
    log_norm = LogNorm(vmin=1.0, vmax=30.0)
    # get variable labels
    varnames = X.columns
    # easier to handle dataframes this way
    X = np.array(X)
    levels =  np.logspace(0, 2, 12)
    get_xlim = fetch_lims(X[:, 0])
    get_ylim = fetch_lims(X[:, 1])

    x_mesh, y_mesh, _ = get_joint_xy(get_xlim, get_ylim)
    concatenate_mesh = np.c_[x_mesh.ravel(), y_mesh.ravel()]
    joint_density = -model.score_samples(concatenate_mesh).reshape(x_mesh.shape)

    plt.contour(x_mesh, y_mesh, joint_density,
                norm=log_norm, levels=levels, colors='slategrey')
    plt.contourf(x_mesh, y_mesh, joint_density,
                 norm=log_norm, levels=levels, cmap='gist_gray')

    joint_density = model.predict(concatenate_mesh).reshape(x_mesh.shape)
    plt.contour(x_mesh, y_mesh, joint_density, colors='tomato')

    plt.scatter(X[:, 0], X[:, 1], marker='.', color='dodgerblue', alpha=.5)
    plt.xlabel(varnames[0]); plt.ylabel(varnames[1])
    plot_centroids(model.means_, model.weights_)

def gmm_information_criteria_report(X_mat, k = np.arange(1, 20), covar_type = ['full', 'tied', 'diag', 'spherical'], random_seed=11238, out="Graph"):
    # Dataframe transposing closure type funct

    tmp_global_aic, tmp_global_bic = [], []
    for i in covar_type:
        tmp_iter_aic, tmp_iter_bic = [], []
        for j in k:
            tmp_model = GaussianMixture(j, covariance_type=i,
                                        random_state = random_seed).fit(X_mat)
            tmp_iter_aic.append(tmp_model.aic(X_mat))
            tmp_iter_bic.append(tmp_model.bic(X_mat))
        tmp_global_aic.append(tmp_iter_aic)
        tmp_global_bic.append(tmp_iter_bic)

    covar_type = covar_type
    tmp_get_aic = handle_df(tmp_global_aic, covar_type)
    tmp_get_bic = handle_df(tmp_global_bic, covar_type)
    tmp_get_aic_max = pd.melt(tmp_get_aic, id_vars=['n_components'],
                              value_vars = covar_type).sort_values(by='value')
    tmp_get_bic_max = pd.melt(tmp_get_bic, id_vars=['n_components'],
                              value_vars = covar_type).sort_values(by='value')
    tmp_top_aic = tmp_get_aic_max.head(3)
    tmp_top_bic = tmp_get_bic_max.head(3)

    if out is "Graph":
        plt.subplot(2, 1, 1)
        for colname, index in tmp_get_aic.drop(columns='n_components').iteritems():
            plt.plot(index, label=colname)
        plt.scatter(tmp_top_aic['n_components'],
                    tmp_top_aic['value'], edgecolors='slategrey',
                    facecolor='none', lw=2, label="Best hyperparams")
        plt.title('Akaike Information Criteria')
        plt.xticks(k - 1, k)
        plt.xlabel('Number of clusters estimated')
        plt.ylabel('AIC')
        plt.legend(loc='center left', bbox_to_anchor=(1, .5))

        plt.subplot(2, 1, 2)
        for colname, index, in tmp_get_bic.drop(columns='n_components').iteritems():
            plt.plot(index, label=colname)
        plt.scatter(tmp_top_bic['n_components'],
                    tmp_top_bic['value'], edgecolors='slategrey',
                    facecolor='none', lw=2, label="Best hyperparams")
        plt.title('Bayesian Information Criteria')
        plt.xticks(k - 1, k)
        plt.xlabel('Number of clusters estimated')
        plt.ylabel('BIC')
        plt.legend(loc='center left', bbox_to_anchor=(1, .5))

    elif out is not "Graph":
        return tmp_get_aic_max, tmp_get_bic_max


def fetch_outliers(model, X, threshold=5):
    """TODO: Docstring for fetch_outliers.

    :model: TODO
    :X: TODO
    :threshold: TODO
    :returns: TODO

    """
    tmp_X = np.array(X)
    extract_densities = model.score_samples(X)
    tmp_threshold = np.percentile(extract_densities, threshold)
    flag_outliers = tmp_X[extract_densities < tmp_threshold]
    plt.scatter(flag_outliers[:, 0], flag_outliers[:, 1], color='orange', marker="s")


from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw ellipses: based on Vanderplas (2017)

    :position: TODO
    :covariance: TODO
    :ax: TODO
    :**kwargs: TODO
    :returns: TODO

    """
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0,0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


# def example_covars():
    # """Draw covariances: drawn from Vanderplas (2017)
    # :returns: TODO

    # """
    # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    # rng = np.random.RandomState(11238)
    # X = np.dot(rng.randn(500, 2), rng.randn(2, 2))

    # for i, covariance in enumerate(['diag', 'spherical', 'full']):
        # model = GMM(1, covariance_type=covariance).fit(X)
        # # ax[i].axis('equal')
        # ax[i].scatter(X[:, 0], X[:, 1], alpha=.5)
        # ax[i].set_xlim(-3, 3)
        # ax[i].set_title('Tipo de covarianza {}'.format(covariance))
        # draw_ellipse(model.means_[0], model.covars_[0], ax[i], alpha=.2)
        # ax[i].xaxis.set_major_formatter(plt.NullFormatter())
        # ax[i].yaxis.set_major_formatter(plt.NullFormatter())
