#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec10_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary files for voting classifiers - ADL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC,NuSVC
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import seaborn as sns

def plot_classification_report(y_test, class_pred, dummy_class=False):
    """TODO: Docstring for plot_classification_report.

    :y_test: TODO
    :class_pred: TODO
    :dummy_class: TODO
    :returns: TODO

    """

    colors = ['dodgerblue', 'tomato']
    report = pd.DataFrame(classification_report(y_test, class_pred, output_dict=True))
    class_specific_values = report.drop(columns=['accuracy', 'macro avg', 'weighted avg'])
    class_specific_values = report.loc[:, class_specific_values.columns].T
    macro_avg = report.drop(index='support')['macro avg']

    for index, value in enumerate(class_specific_values.index):
        plt.scatter(class_specific_values['precision'][value], [1], marker='x', c=colors[index])
        plt.scatter(class_specific_values['recall'][value], [2], marker='x', c=colors[index])
        plt.scatter(class_specific_values['f1-score'][value], [3], marker='x', c=colors[index], label=f"Class: {index}")

    plt.scatter(macro_avg, [1, 2, 3], color='forestgreen', label='Macro Average')
    plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'F1-Score'])

    if dummy_class is True:
        plt.axvline(.5, label = '.5 Boundary', linestyle='--')

    # plt.legend(loc='center left', bbox_to_anchor=(1, .5))



def voting_classifiers_behavior(predictions, votes_n, majority):
    """TODO: Docstring for voting_classifiers_behavior.

    :predictions: TODO
    :votes_n: TODO
    :majority: TODO
    :returns: TODO

    """
    # iniciamos un contenedor vacío
    tmp_holder = pd.DataFrame()
    # buscamos el cruce entre cada predicción existente a nivel de modelo
    for i in np.unique(predictions[votes_n]):
        # y la predicción a nivel de comité
        for j in np.unique(predictions[majority]):
            # separamos los casos que cumplan con ambas condiciones
            tmp_subset = predictions[np.logical_and(
                predictions[votes_n] == i,
                predictions[majority] == j
            )]
            # extraemos la cantidad de casos existentes
            tmp_rows_n = tmp_subset.shape[0]
            # Si la cantidad de casos existentes es mayor a cero
            if tmp_rows_n > 0:
                # registramos la importancia del clasificador RESPECTO A LA CANTIDAD DE CASOS EXISTENTES.
                tmp_holder[f'Votes: {i} / Class: {j}'] = round(tmp_subset.apply(sum) / tmp_rows_n, 3)
    # transposicionar
    tmp_holder = tmp_holder.T
    # Eliminamos columnas redundantes del dataframe
    tmp_holder = tmp_holder.drop(columns=[votes_n, majority])
    # visualizamos la matriz resultante
    sns.heatmap(tmp_holder, annot=True, cmap='coolwarm_r', cbar=False)


def plot_describe_variables(df, rows, cols):
    """TODO: Docstring for plot_describe_variables.
    :returns: TODO

    """
    for index, (colname, serie) in enumerate(df.iteritems()):
        plt.subplot(rows, cols, index + 1)
        if serie.dtype == 'object':
            sns.countplot(serie)
            plt.axhline(serie.value_counts().mean())
            plt.title(colname)
        else:
            sns.distplot(serie)
            plt.axvline(np.mean(serie))
            plt.title(colname)
    plt.tight_layout()


def weighting_schedule(voting_ensemble, X_train, X_test, y_train, y_test, weights_dict, plot_scheme=True, plot_performance=True):
    """TODO: Docstring for weighting_schedule.

    :voting_ensemble: TODO
    :X_train: TODO
    :X_test: TODO
    :y_train: TODO
    :y_test: TODO
    :weights_dict: TODO
    :plot_scheme: TODO
    :plot_performance: TODO
    :returns: TODO

    """

    def weight_scheme():
        """TODO: Docstring for weight_scheme.
        :returns: TODO

        """
        weights = pd.DataFrame(weights_dict)
        weights['model'] = [i[0] for i in voting_ensemble.estimators]
        weights = weights.set_index('model')
        sns.heatmap(weights, annot=True, cmap='Blues', cbar=False)
        plt.title('Esquema de Ponderación')

    def weight_performance():
        """TODO: Docstring for weight_performance.
        :returns: TODO

        """

        n_scheme = len(weights_dict)
        f1_metrics, accuracy = [], []
        f1_metrics_train, accuracy_train = [], []

        for i in weights_dict:
            model = voting_ensemble.set_params(weights=weights_dict[i]).fit(X_train, y_train)
            tmp_model_yhat = model.predict(X_test)
            tmp_model_yhat_train = model.predict(X_train)
            f1_metrics.append(f1_score(y_test, tmp_model_yhat).round(3))
            f1_metrics_train.append(f1_score(y_train, tmp_model_yhat_train).round(3))
            accuracy.append(accuracy_score(y_test, tmp_model_yhat).round(3))
            accuracy_train.append(accuracy_score(y_train, tmp_model_yhat_train).round(3))
        plt.plot(range(n_scheme), accuracy, 'o', color='tomato', alpha=.5, label='Exactitud-Test')
        plt.plot(range(n_scheme), f1_metrics, 'x', color='tomato', alpha=.5, label='F1-Test')
        plt.plot(range(n_scheme), accuracy_train, 'o', color='dodgerblue', alpha=.5, label='Exactitud-Train')
        plt.plot(range(n_scheme), f1_metrics_train, 'x', color='dodgerblue', alpha=.5, label='F1-Train')
        plt.xticks(ticks=range(n_scheme), labels=list(weights_dict.keys()), rotation=90)
        plt.title('Desempeño en Train/Test')
        plt.legend(loc='center left', bbox_to_anchor=(1, .5))


    if plot_scheme is True and plot_performance is True:
        plt.subplot(1, 2, 1)
        weight_scheme()
        plt.subplot(1, 2, 2)
        weight_performance()

    else:
        if plot_scheme is True:
            weight_scheme()
        elif plot_performance is True:
            weight_performance()

def committee_voting(voting_ensemble, X_train, X_test, y_train, y_test):
    """TODO: Docstring for committee_voting.

    :voting_ensemble: TODO
    :returns: TODO

    """
    # iniciar dataframe vacio para guardar valores
    individual_preds = pd.DataFrame()
    # preservamos la lista de tuplas
    voting_estimators = voting_ensemble.estimators
    # para cada iterador en la lista de tuplas
    for i in voting_estimators:
        # generamos la estimación específica
        individual_preds[i[0]] = i[1].fit(X_train, y_train).predict(X_test)
    # extraemos los votos individuales de cada clasificador
    individual_preds['votes_n'] = individual_preds.loc[:, voting_estimators[0][0]:voting_estimators[-1][0]].apply(np.sum, axis=1)
    # generamos la predicción del ensamble heterogéneo
    individual_preds['Majority'] = voting_ensemble.set_params(weights=None).predict(X_test)

    # iniciamos un contenedor vacío
    tmp_holder = pd.DataFrame()
    # buscamos el cruce entre cada predicción existente a nivel de modelo
    for i in np.unique(individual_preds['votes_n']):
        # y la predicción a nivel de comité
        for j in np.unique(individual_preds['Majority']):
            # separamos los casos que cumplan con ambas condiciones
            tmp_subset = individual_preds[np.logical_and(
                individual_preds['votes_n'] == i,
                individual_preds['Majority'] == j
            )]
            # extraemos la cantidad de casos existentes
            tmp_rows_n = tmp_subset.shape[0]
            # Si la cantidad de casos existentes es mayor a cero
            if tmp_rows_n > 0:
                # registramos la importancia del clasificador RESPECTO A LA CANTIDAD DE CASOS EXISTENTES.
                tmp_holder[f'Votes: {i} / Class: {j}'] = round(tmp_subset.apply(sum) / tmp_rows_n, 3)
    # transpose
    tmp_holder = tmp_holder.T
    # Eliminamos columnas redundantes del dataframe
    tmp_holder = tmp_holder.drop(columns=['votes_n', 'Majority'])
    # visualizamos la matriz resultante
    sns.heatmap(tmp_holder, annot=True, cmap='coolwarm_r', cbar=False)

def annotated_barplot(variable):
    """TODO: Docstring for annotated_barplot.

    :variable: TODO
    :returns: TODO

    """
    # extraemos la frecuencia de ocurrencia 
    tmp_values = variable.value_counts('%')
    # graficamos y preservamos los atributos del gráfico
    tmp_ax = tmp_values.plot(kind='bar')
    # para cada patch en el gráfico
    for index, patch in enumerate(tmp_ax.patches):
        # anotamos la frecuencia porcentual
        tmp_ax.annotate(tmp_values[index].round(3),
                        # al centro de la barra y arriba
                        xy=(patch.get_x() + .125, patch.get_height() + 0.01),
                        # inferimos el color de cada barra
                        color=patch.get_facecolor())

