#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec5_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary files for Support Vector Machine Lectures - adl
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import seaborn as sns
from scipy import stats

color_palette_sequential = [ '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016c59', '#014636']
fetch_lims = lambda x: [np.floor(np.min(x)), np.ceil(np.max(x))]

def get_joint_xy(xlim, ylim):
    """TODO: Given two lists, compute coordinates and transpose to a new list

    :xlim: first array
    :ylim: second array
    :returns: A tuple with each marginal and the joint values

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

def setup_svm_problem():
    """Setup the linear candidates and maximum margin classifier
    :returns: a matplotlib figure

    """
    plt.subplot(1, 2, 1)
    X, y = make_blobs(random_state=754, cluster_std=1, centers=2)
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], marker='s', color='dodgerblue', alpha=.5)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], marker='o', color='tomato', alpha=.5)
    get_xlim = plt.xlim()
    get_ylim = plt.ylim()

    x_range = np.linspace(get_xlim[0], get_xlim[1])

    for slope, intercept in [(1, -8), (.25, -5.1), (-.3, -3), (6, -15)]:
        plt.plot(x_range, slope * x_range + intercept, color='slategrey')
        plt.ylim(get_ylim)
        plt.xlim(get_xlim)
    plt.title('Clasificadores lineares candidatos')

    plt.subplot(1, 2, 2)
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], marker='s', color='dodgerblue', alpha=.5)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], marker='o', color='tomato', alpha=.5)
    model = SVC(kernel='linear', C=1).fit(X, y)

    tmp_y, tmp_x = np.meshgrid(np.linspace(get_ylim[0], get_ylim[1]),
                               np.linspace(get_xlim[0], get_xlim[1]))
    joint_xy = np.vstack([tmp_x.ravel(), tmp_y.ravel()]).T
    tmp_z = model.decision_function(joint_xy).reshape(tmp_x.shape)

    plt.contour(tmp_x, tmp_y, tmp_z, colors='slategrey', levels=[-1, 0, 1], linestyles=['--','-','--'])
    plt.scatter(model.support_vectors_[:,0],
                model.support_vectors_[:, 1],
                s=250, linewidth=2, facecolor='none',
                edgecolor='slategrey')
    plt.title('Clasificador del máximo márgen')
    plt.text(7.2, -2, r'$M=-1$')
    plt.text(7.2, -4.9, r'$M=1$')
    plt.text(7.2, -3.5, r'$\beta^{T} + \beta_{0}$', size=16)

def plot_cv_grid_search(model,X_mat, y_vec, param1, param1_range, param2, param2_range,cv=10):
    """Report best hyperparameters

    :model: A sklearn class
    :X_mat: Feature maxtrix
    :y_vec: target vector
    :param1: Hyperparam name. Must comply with a valid sklearn param
    :param1_range: Hyperparameter range
    :param2: Hyperparam name. Must comply with a valid sklearn param
    :param2_range: Hyperparameter range
    :returns: A boxplot graph for each combination, and a heatmap signaling the best parameter

    """
    # init storing vars
    tmp_dict = {}
    best_score_signaler = 0
    param_1_holder = []

    # init loop
    for i in param1_range:
        param_2_holder=[]

        # init second loop
        for j in param2_range:

            # crossvalidate entry params
            tmp_model = cross_val_score(model.set_params(param1=i, param2=j), X_mat, y_vec, cv=cv)
            # get params combination to strong
            param_mixin = "{0}: {2}, {1}: {3}".format(param1, param2, i, j)
            # store params to a dict
            tmp_dict[param_mixin] = list(tmp_model)
            # store mean to param2 holder
            param_2_holder.append(np.mean(tmp_model))
            # fetch signaler
            if np.mean(tmp_model) > best_score_signaler:
                best_score_signaler=np.mean(tmp_model)
                best_params = param_mixin
        # append param2 array to param1 array
        param_1_holder.append(param_2_holder)
    # reshape to a matrix
    cv_params = np.array(param_1_holder).reshape(len(param1_range), len(param2_range))
    # init grid
    grid = GridSpec(1, 2, width_ratios=[2, 1])
    plt.subplot(grid[0])
    # cv scores boxplot
    plt.boxplot(tmp_dict.values(), showmeans=True);
    plt.xticks(range(1, len(tmp_dict.values())+ 1), tmp_dict.keys());
    plt.title("Best params (on {} CV) \n {}".format(cv, best_params));
    # performance metric on matrix
    plt.subplot(grid[1])
    sns.heatmap(cv_params, annot=True, cmap='Blues',
                cbar=False, xticklabels=param2_range, yticklabels=param1_range)
    plt.xlabel(param2)
    plt.ylabel(param1)

def loss_functions():
    """TODO: Docstring for loss_functions.
    :returns: TODO

    """
    def huber_loss(y_true, y_pred):
        """TODO: Docstring for huber_loss.

        :y_true: TODO
        :y_pred: TODO
        :returns: TODO

        """
        z = y_pred * y_true
        loss = -4 * z
        loss[z >= -1] = (1 - z[z >= -1]) ** 2
        loss[z >= 1] = 0
        return loss

    x_axis = np.linspace(-4, 4, 100)

    plt.plot([-4, 0, 0, 4], [1, 1, 0, 0], label="Sharp", lw=3)
    plt.plot(x_axis, np.where(x_axis < 1, 1 - x_axis, 0), label='Hinge', lw = 3)
    plt.plot(x_axis, x_axis ** 2, label='Squared', lw=3)
    plt.plot(x_axis, np.where(x_axis < 1, 1 - x_axis, 0) ** 2, label='Squared hinge', lw = 3)
    plt.plot(x_axis, huber_loss(x_axis, 1), label='Huber', lw=3)
    plt.ylim(0, 10)
    plt.legend()
    plt.title('Funciones de pérdida')

def extract_binary_target_features(df, target, k=5,threshold=0.01, graph=False, abs_corr=False):
    """Report best features given point biserial correlation

    :df: A pandas dataframe object
    :target: Target vector contained in the dataframe
    :k:  parameters to preserve.
    :threshold: a p-value critical region
    :graph: if True, function will return a dataframe with variable name, correlation and p-value
    :abs_corr: if True, correlations reported will be on the absolute.
    :returns: TODO

    """
    hold_names, hold_coefs, hold_pval = [], [], []

    for colname, serie in df.iteritems():
        if colname != target:
            hold_names.append(colname)
            biserial_rho = stats.pointbiserialr(serie, df[target])
            if abs_corr is True:
                hold_coefs.append(np.abs(biserial_rho[0]))
            else:
                hold_coefs.append(biserial_rho[0])
            hold_pval.append(biserial_rho[1])

    feats_df = pd.DataFrame({
        'var':hold_names,
        'corrs': np.round(hold_coefs, 3),
        'pval': hold_pval
    })
    feats_df = feats_df.set_index('var')
    feats_df = feats_df.sort_values(by='corrs', ascending=False)
    feats_df = feats_df[:k]

    if graph is True:
        plt.barh(feats_df.index,
                 feats_df['corrs'],
                 color=np.where(feats_df['pval'] < threshold, 'dodgerblue', 'tomato'))
    else:
        return feats_df

def slack_variables():
    """TODO: Docstring for slack_variables.
    :returns: TODO

    """
    X, y = make_blobs(random_state=754, cluster_std=2, centers=2)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], markers='s', color='dodgerblue')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], markers='o', color='tomato')
    get_xlim = plt.xlim()
    get_ylim = plt.ylim()
    model = SVC(kernel='linear').fit(X, y)
    tmp_y, tmp_x = np.meshgrid(np.linspace(get_ylim[0], get_ylim[1]),
                               np.linspace(get_xlim[0], get_ylim[1]))
    joint_xy = np.vstack([tmp_x.ravel(), tmp_y.ravel()]).T
    tmp_z = model.decision_function(joint_xy).reshape(tmp_x.shape)
    plt.contour(tmp_x, tmp_y, tmp_z, colors='slategrey', levels=[-1, 0, 1],
                alpha=.5, linestyles=['--', '-','--'],
                linewidths=[1, 3, 1])
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=300, linewidth=1, facecolor='none', edgecolor='slategrey')


def svm_c_hyperparameter(X, y, c_range = [0.0001, 0.1, 1000]):
    """TODO: Docstring for svm_c_hyperparameter.

    :X: TODO
    :y: TODO
    :returns: TODO

    """
    get_xlim = fetch_lims(X[:, 0])
    get_ylim = fetch_lims(X[:, 1])
    x_mesh, y_mesh, joint_xy = get_joint_xy(get_xlim, get_ylim)

    for index, c in enumerate(c_range):
        if len(c_range) > 3:
            plt.subplot(2, 3, index + 1)
        else:
            plt.subplot(1, 3, index + 1)
        tmp_model = SVC(kernel = 'rbf', C=c, gamma=.01).fit(X, y)
        tmp_densities = tmp_model.decision_function(joint_xy).reshape(x_mesh.shape)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='orange', alpha=.8, s=25, marker='s')
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='dodgerblue', alpha=.8, s=25, marker='o')
        plt.contourf(x_mesh, y_mesh, tmp_densities, cmap='Greys', alpha=.5)
        plt.title("C: {}".format(c), fontsize=18)
        plt.tight_layout()

def svm_logical_xor_data(nsize=400, random_state=11238):
    """TODO: Docstring for svm_demo_data.

    :nsize: TODO
    :random_state: TODO
    :returns: TODO

    """
    np.random.seed(random_state)
    x_xor = np.random.randn(nsize, 2)
    y_xor = np.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    return x_xor, y_xor

def svm_gamma_hyperparameter(X, y, gamma_range=[0.0001, 0.1, 1000]):
    """TODO: Docstring for svm_c_hyperparameter.

    :X: TODO
    :y: TODO
    :returns: TODO

    """
    get_xlim = fetch_lims(X[:, 0])
    get_ylim = fetch_lims(X[:, 1])
    x_mesh, y_mesh, joint_xy = get_joint_xy(get_xlim, get_ylim)

    for index, g in enumerate(gamma_range):
        if len(gamma_range) > 3:
            plt.subplot(2, 3, index + 1)
        else:
            plt.subplot(1, 3, index + 1)
        tmp_model = SVC(kernel = 'rbf', gamma=g, C=1).fit(X, y)
        tmp_densities = tmp_model.decision_function(joint_xy).reshape(x_mesh.shape)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='orange', alpha=.8, s=25, marker='s')
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='dodgerblue', alpha=.8, s=25, marker='o')
        plt.contourf(x_mesh, y_mesh, tmp_densities, cmap='Greys', alpha=.5)
        plt.title("Gamma: {}".format(g), fontsize=18)
        plt.tight_layout()

def svm_non_separable(plot_slacks = False, plot_xi=False):
    """TODO: Docstring for svm_non_separable.
    :returns: TODO

    """
    X, y = make_blobs(random_state=754, cluster_std=1.5, centers=2, n_samples=50)
    get_xlim = fetch_lims(X[:, 0])
    get_ylim = fetch_lims(X[:, 1])
    x_mesh, y_mesh, joint_xy = get_joint_xy(get_xlim, get_ylim)

    tmp_model = SVC(kernel='linear', C=0.01).fit(X, y)
    tmp_densities = tmp_model.decision_function(joint_xy).reshape(x_mesh.shape)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='s', color='dodgerblue')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o', color='tomato')
    plt.contour(x_mesh, y_mesh, tmp_densities,
                colors='slategrey', levels=[-1, 0, 1],
                alpha=.5, linestyles=['--','-','--'],
                linewidths=[1, 3, 1])
    if plot_slacks is True:
        sv = tmp_model.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], s=300, linewidth=1,
                    facecolor='none', edgecolor='slategrey')
    if plot_xi is True:
        plt.text(.5, -5.5, r'$\xi < 1$', fontsize=14, color='dodgerblue')
        plt.scatter(2.2, -2, color='dodgerblue', marker='s')
        plt.scatter(2.2, -2, facecolor='none', edgecolor='slategrey', linewidth=1,s=300)
        plt.text(1, -2.3, r'$\xi > 1$', fontsize=14, color='dodgerblue')
def plot_class_report(y_test, y_hat, classes_labels):
    """TODO: Docstring for plot_class_report.

    :y_test: TODO
    :y_hat: TODO
    :classes_labels: TODO
    :returns: TODO

    """
    tmp_report = classification_report(y_test, y_hat, output_dict=True)
    targets = list(classes_labels)
    targets.append('average')
    tmp_report = pd.DataFrame(tmp_report)\
                                .drop(columns=['micro avg', 'macro avg'])
    tmp_report.columns = targets
    tmp_report = tmp_report.drop(labels='support')
    tmp_report = tmp_report.drop(columns='average')
    tmp_report = tmp_report.T

    for index, (colname, serie) in enumerate(tmp_report.iteritems()):
        plt.subplot(3, 1, index + 1)
        serie.plot(kind = 'barh')
        plt.title(f"Métrica: {colname}")
        plt.tight_layout()
