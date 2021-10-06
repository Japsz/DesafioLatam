#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, mean_squared_error, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve
import seaborn as sns
from scipy import stats


###################################
#  Fixed Figures for replication  #
###################################




def multivariate_graph(mu_vec=[0,0], sigma_mat = [[1, .9], [.9, 1]], var_range =3):
    X_mat, y_vec = np.mgrid[-var_range:var_range:.01, -var_range:var_range:.01]
    tmp_position = np.dstack((X_mat, y_vec))
    tmp_multivariate = stats.multivariate_normal(mu_vec, sigma_mat)
    plt.contourf(X_mat, y_vec, tmp_multivariate.pdf(tmp_position), cmap='Blues')
    plt.title(r'$\mu_{1}=$' + "{} ".format(str(mu_vec[0])) +
              r'$\mu_{2}=$' + "{}; ".format(str(mu_vec[1])) +
              r'$\sigma_{1}= $' + "{} ".format(str(sigma_mat[0])) +
              r'$\sigma_{2}= $' + "{} ".format(str(sigma_mat[1])))


def plot_mvn():
    plt.subplot(1, 3, 1)
    multivariate_graph(sigma_mat=[[1, -.5], [-.5, 1]])
    plt.text(-2.5, 2.5, 'a', fontsize=18, fontweight='bold')
    plt.subplot(1, 3, 2)
    multivariate_graph(sigma_mat=[[1,0], [0,1]])
    plt.text(-2.5, 2.5, 'b', fontsize=18, fontweight='bold')
    plt.subplot(1, 3, 3)
    multivariate_graph(sigma_mat=[[1, .9], [.9, 1]])
    plt.text(-2.5, 2.5, 'c', fontsize=18, fontweight='bold')


def plot_lda():
    """TODO: Docstring for plot_lda.
    :returns: TODO

    """
    X_mat, y_vec = np.mgrid[-4:4:.01, -4:4:.01]
    tmp_position = np.dstack((X_mat, y_vec))
    tmp_multivariate_1 = stats.multivariate_normal([-1.5, -1], [[1, 0], [0, 1]])
    tmp_multivariate_2 = stats.multivariate_normal([2.5, 1], [[1, 0], [0, 1]])
    plt.contour(X_mat, y_vec, tmp_multivariate_1.pdf(tmp_position), cmap='Reds')
    plt.text(-3, 3, r'$x\vert y = 0 \sim MultivarNorm(\mu_{0}, \Sigma)$', color='tomato', fontsize=16, weight='bold')
    plt.plot(-1.5, -1, color='tomato',marker='s', markersize=12, label=r'$\mu \vert y = 0$')
    x, y = np.random.multivariate_normal([-1.5, -1], [[2, 0], [0, 2]], size=100).T
    plt.plot(x, y, 'o', color='tomato', alpha=.3)
    plt.contour(X_mat, y_vec, tmp_multivariate_2.pdf(tmp_position), cmap='Blues')
    plt.plot(2.5, 1, color='dodgerblue',marker='s', markersize=12, label=r'$\mu \vert y = 1$')
    plt.text(1, -3, r'$x \vert y = 1 \sim MultivarNorm(\mu_{1}, \Sigma)$', color='dodgerblue', fontsize=16, weight='bold')
    x, y = np.random.multivariate_normal([2.5, 1], [[2, 0], [0, 2]], size=100).T
    plt.plot(x, y, 'o', color='dodgerblue', alpha=.3)
    plt.plot([-1.5, 2.5], [-1, 1], color='purple', linestyle='--', lw=4)
    plt.axis([-4, 4, -4, 4])

def plot_densities():
    np.random.seed(11238)
    x_axis_1 = np.linspace(0.5, -4, 100)
    x_axis_2 = np.linspace(0.5, 4, 100)
    norm_1 = np.random.normal(loc=-1.5, scale=.5, size=100)
    density_1 = stats.kde.gaussian_kde(np.hstack((x_axis_1, norm_1)))
    norm_2 = np.random.normal(loc=2.5, scale=.5, size=100)
    density_2 = stats.kde.gaussian_kde(np.hstack((x_axis_2, norm_2)))
    plt.plot(norm_1, [0.05]*len(norm_1), '|', color='tomato')
    plt.plot(x_axis_1, density_1(x_axis_1), color='tomato', lw=4)
    plt.plot(norm_2, [0.05]*len(norm_2), '|', color='dodgerblue')
    plt.plot(x_axis_2, density_2(x_axis_2), color='dodgerblue', lw=4)
    plt.axvline(.5, color='slategrey', lw=4, linestyle='--', label='Minimizador')
    plt.axhline(y=-.01, xmin=.30, xmax=.80, color='purple', linestyle='--', lw=6, label='Nueva proyección de datos')
    plt.legend()
    plt.xlim(-4, 4)

def plot_features():
    """TODO: Docstring for plot_features.
    :returns: TODO

    """
    np.random.seed(11238)
    x_1, y_1 = np.random.multivariate_normal([-1.5, -1], [[2, 0], [0, 2]], size=100).T
    x_2, y_2 = np.random.multivariate_normal([2.5, 1], [[2, 0], [0, 2]], size=100).T
    plt.plot(x_2, y_2, 'o', color='dodgerblue', alpha=.5)
    plt.axhline(1, xmin=.81, xmax=1, color='dodgerblue', lw=3)
    plt.axvline(2.5, ymin=0, ymax=.57, color='dodgerblue', lw=3)
    plt.plot(2.5, 1, color='dodgerblue',marker='s', markersize=12, label=r'$\mu \vert y = 1$')
    plt.plot(x_1, y_1,'o',  color='tomato', alpha=.5)
    plt.axhline(-1,xmin=0, xmax=.31,  color='tomato', lw=3)
    plt.axvline(-1.5,ymin=0, ymax=.36,  color='tomato',lw=3)
    plt.plot(-1.5, -1, color='tomato',marker='s', markersize=12, label=r'$\mu \vert y = 0$')
    plt.legend()

def plot_lda_sequence():
    """TODO: Docstring for plot_lda_sequence.
    :returns: TODO

    """
    grid = GridSpec(3,1, height_ratios=[2, 3, 1])
    plt.subplot(grid[0])
    plot_features()
    plt.title('1 - Matriz de Datos', fontsize=16)
    plt.xlim(-4, 4)
    plt.subplot(grid[1])
    plot_lda()
    plt.title('2 - Clasificación', fontsize=16)
    plt.subplot(grid[2])
    plot_densities()
    plt.title('3 - Densidades de Clase Inferidas', fontsize=16)



#######################
#  Diagnosis figures  #
#######################


# custom colors

color_palette_divergent = LinearSegmentedColormap.from_list('ee', ['#E27872', '#F9F9F8', '#509A9A'])
color_palette_discrete = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']
color_palette_sequential = [ '#ece3f0', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016c59', '#014636']

markers = ['o', '^', '*','H', 'P', 'D', 'X', 'h', 'p', 'd', 'c']

def generate_mesh_grid(df, x1, x2):
    """TODO: Docstring for generate_mesh_grid.

    :df: TODO
    :x1: TODO
    :x2: TODO
    :returns: TODO

    """

    # a partir de dos columnas separadas
    tmp_x = df.loc[:, [x1, x2]]
    # retornar una red con puntos ij
    x_0, x_1 = np.meshgrid(
        # considerando el mínimo y máximo de x1, simulado 100 y reescalado entre -1 y 1
        np.linspace(np.min(tmp_x[x1]), np.max(tmp_x[x1]), num=100).reshape(-1, 1),
        # considerando el mínimo y máximo de x2, simulado 100 y reescalado entre -1 y 1
        np.linspace(np.min(tmp_x[x2]), np.max(tmp_x[x2]), num=100).reshape(-1, 1)
    )
    return x_0, x_1

def probability_contours(model,df,target, x1, x2,classes_labels, fill_contours=False):
    """TODO: Docstring for probability_contours.

    :model: TODO
    :df: TODO
    :target: TODO
    :x1: TODO
    :x2: TODO
    :fill_contours: TODO
    :returns: TODO

    """
    # a partir de dos columnas separadas
    tmp_x = df.loc[:, [x1, x2]]
    # estimar un modelo en base a las columnas y el vector objetivo
    tmp_model = model.fit(tmp_x, target)
    # extraemos los puntos ij
    x_0, x_1 = generate_mesh_grid(tmp_x, x1, x2)
    # Aplanamos cada punto ij, y los concatenamos
    map_x = np.c_[x_0.ravel(), x_1.ravel()]
    # implementamos la predicción de Pr(y)
    predict_y_pr = tmp_model.predict_proba(map_x)
    # implementamos la predicción de la clase con argmax
    predict_y = tmp_model.predict(map_x)
    # extraemos los límites de probabilidad
    boundaries_pr = predict_y_pr[:, 1].reshape(x_1.shape)
    # extraemos las clases
    boundaries_y = predict_y.reshape(x_0.shape)
    custom_cmap = ListedColormap(color_palette_sequential)

    # por cada clase estimable
    for i in target.unique():
        # graficamos los puntos correspondientes en las dos columnas
        plt.plot(tmp_x[target == i][x1], tmp_x[target == i][x2],
                 '.', marker=markers[i], color=color_palette_discrete[i],
                 label = "{}".format(classes_labels[i]), alpha=.8)
    # Graficamos los límites
    if fill_contours is True:
        custom_cmap = LinearSegmentedColormap.from_list('lista', color_palette_sequential)
        plt.contourf(x_0, x_1, boundaries_pr, cmap=custom_cmap)
        plt.colorbar()
        plt.clim(0, 1)
    else:
        vis_boundaries = plt.contour(x_0, x_1, boundaries_pr, cmap = custom_cmap)
        plt.clabel(vis_boundaries, inline=1)

    plt.legend(framealpha=0.5, edgecolor='slategrey', fancybox=True)
    plt.xlabel(x1)
    plt.ylabel(x2)

def plot_confusion_matrix(y_test, y_hat, classes_labels):
    """TODO: Docstring for plot_confusion_matrix.

    :y_test: TODO
    :y_hat: TODO
    :returns: TODO

    """
    tmp_confused = confusion_matrix(y_test, y_hat)
    custom_cmap = LinearSegmentedColormap.from_list('lista', color_palette_sequential)
    sns.heatmap(tmp_confused, annot=True, cbar=False, cmap=custom_cmap, xticklabels=classes_labels,
                yticklabels=classes_labels)
    plt.xlabel('Classes on testing data')
    plt.ylabel('Predicted classes on training')
    plt.grid(False)



def plot_class_report(y_test, y_hat, classes_labels):
    """TODO: Docstring for plot_class_report.

    :y_test: TODO
    :y_Hat: TODO
    :classes_labels: TODO
    :returns: TODO

    """
    tmp_report = classification_report(y_test, y_hat, output_dict=True)
    targets = list(classes_labels)
    targets.append('average')
    tmp_report = pd.DataFrame(tmp_report)\
                    .drop(columns=['weighted avg', 'macro avg'])
    tmp_report.columns = targets
    tmp_report = tmp_report.drop(labels='support')
    tmp_report = tmp_report.drop(columns='average')
    tmp_report = tmp_report.T

    for index, (colname, serie) in enumerate(tmp_report.iteritems()):
        plt.subplot(3, 1, index + 1)
        serie.plot(kind = 'barh')
        plt.title(f"Métrica: {colname}")
        plt.tight_layout()



