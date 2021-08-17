#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec4_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary file for intro to data science - adl
"""

from collections import Counter
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import t


colors = ["tomato", "darkgoldenrod", "limegreen", "dodgerblue", "sienna", "slategray"]

def generate_corr_matrix(rho_params = np.linspace(-1.0, 1.0, 20).round(1)):
    """docstring for generate_corr_matrix"""
    for i, corr in enumerate(rho_params):
        plt.subplot(4, 5, i+1)
        np.random.seed(666)
        x, y = np.random.multivariate_normal(mean = [0, 0], cov = [[1, corr], [corr, 1]], size = 100).T
        beta_1, beta_0 = np.polyfit(x, y, 1)
        plt.plot(x, y, 'o', alpha=.5)
        plt.plot(x, [beta_1 * j + beta_0 for j in x], 'b', color='tomato')
        plt.axis('off')
        plt.title(r'$\rho$={}'.format(corr), fontweight='bold')

def law_large_numbers(function = np.random.poisson, sample_size = 2000, Theta = 10):
    np.random.seed(2)
    for i in range(len(colors)):
        sample = function(Theta, size=sample_size)
        x_span = range(1, sample_size, 100)
        sample_average = [sample[:j].mean() for j in x_span]
        plt.plot(x_span, sample_average, lw=1.5, label = r'$\hat\theta$ en Ensayo {}'.format(i+1),
                 color = colors[i], linestyle='--')

    plt.title('Medias muestrales y tamaño muestral')
    plt.ylabel('Media muestral')
    plt.xlabel('Tamaño muestral')
    plt.axhline(Theta, lw=3)
    plt.annotate(r'$\Theta$', xy = (sample_size - 100, 10.2), fontsize =20, color='#1c6cab')
    plt.legend()

def fdp_normal(x, mu = 0, sigma =1):
    bracket_exponencial = np.exp(-(x - mu) ** 2/ (2 * sigma ** 2))
    frac = np.sqrt(2 * np.pi) * sigma
    return (frac ** -1) * bracket_exponencial

def fdc_normal(x, mu=0, sigma=1):
    elemental = 1 + math.erf((x - mu) / np.sqrt(2) / sigma)
    return elemental / 2

def bernoulli(p):
    # genera 1 si el número aleatorio es mayor a p
    return 1 if np.random.random() < p else 0

def binomial(n, p):
    # genera la suma de ensayos de bernoulli con p probabilidad repetido n veces
    return sum(bernoulli(p) for _ in range(n))

def plot_hist(p, n, points):
    # genera un array temporal para guardar distribuciones binomiales repetidas `points` veces
    tmp = [binomial(n, p) for _ in range(points)]
    # contador de instancias
    hist = Counter(tmp)
    # delimitador de ancho de columnas en histograma
    bins = [x -0.4 for x in hist.keys()]
    # estimador de densidad
    density = [v / points for v in hist.values()]
    plt.bar(bins, density, color='dodgerblue',alpha=.5)

    # guardamos la media y la desviación estandar
    mu = p * n
    sigma = np.sqrt(mu * (1 - p))

    # declaramos un eje x con un rango igual a la cantidad de elementos en `tmp`
    xaxis = range(min(tmp), max(tmp) + 1)
    # generar la distribución normal a partir de intervalos inferiores y superiores
    yaxis = [fdc_normal(i + 0.5, mu, sigma) - fdc_normal(i - 0.5, mu, sigma) for i in xaxis]
    # graficar
    plt.plot(xaxis, yaxis, color='tomato')
    # señalar la media
    plt.axvline(mu, color='#1c6cab', lw=2, linestyle='--')
    plt.title("Iteración: " + str(points), fontsize=10)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

def central_limit_theorem(values = [1, 5, 10, 100, 1000, 10000]):
    for i, v in enumerate(values):
        plt.subplot(2, 3, i+1)
        plot_hist(.2, 1000, v)

def confidence_intervals():
    sims = 100
    coverage = np.empty(sims) # 1 si contiene theta; 0 de lo contrario
    lower_bound = np.empty(sims) # intervalo inferior
    upper_bound = np.empty(sims) # intervalo superior

    # por cada elemento entre 1 y 100
    for i in range(sims):
        # generemos una distribución X˜N(0,1)
        draws = np.random.normal(loc=0, scale=math.sqrt(1), size=500)
        # calculemos el intervalo inferior al 95 %
        lower_bound[i] = draws.mean() - (draws.std()/math.sqrt(500)) * 1.96
        # calculemos el intervalo superior al 95 %
        upper_bound[i] = draws.mean() + (draws.std()/math.sqrt(500)) * 1.96
        # Si entre los intervalos no se contiene a 0
        if (lower_bound[i] < 0) and (upper_bound[i] > 0):
            # marcar como 1
            coverage[i] = 1
            # de lo contrario
        else:
            # marcar como 0
            coverage[i] = 0

    cnt = []
    for i, ci in enumerate(zip(lower_bound, upper_bound)):
        cnt.append(i+1)

    # guardemos la información en un dataframe para facilitar la manipulación
    coverage_range = pd.DataFrame({ 'counter': cnt,
                                   'lb': lower_bound,
                                   'ub': upper_bound,
                                   'rejected' : coverage} )

    plt.axhline(y = 0, lw=3, color = '#1c6cab')
    plt.annotate(r'$\theta$ ', xy=(101,.01), fontsize=20, color='#1c6cab')

    for i, row in coverage_range.iterrows():
        if row['rejected'] == 1:
            plt.vlines(row['counter'], row['lb'], row['ub'], color = 'dodgerblue', linewidth = 1.5)
        else:
            plt.vlines(row['counter'], row['lb'], row['ub'], color = 'tomato', linewidth = 3)

    plt.xlabel('Iteraciones')
    plt.ylabel('Parámetro')
    plt.title('')

def significance_threshold(cutoff, c):
    xaxis = np.linspace(-3, 3, 500)
    t_distribution = stats.t.pdf(xaxis, 500)
    cutoff_point = stats.t.ppf(cutoff, 500)
    plt.plot(xaxis, t_distribution, color='#1c6cab', lw=3)
    plt.axvline(cutoff_point, 0, 0.4, color=c, 
                label=r'Sig: {0}% z: {1}'.format(int((1-(2*cutoff)) * 100), -round(cutoff_point, 2)),
                linestyle='--')
    plt.annotate("{}".format(-cutoff), xy=(cutoff_point-.25, 0.16), color=c, )
    plt.axvline(-cutoff_point, 0, 0.4, color=c, linestyle='--')
    plt.annotate("{}".format(cutoff), xy=(-cutoff_point, 0.16), color=c)
    plt.annotate("Falla en Rechazar \nHipótesis Nula", xy=(-0.35,.25))
    plt.fill_between(xaxis, 0, .4, where=xaxis > 1.62, alpha=.1, facecolor='slategrey')
    plt.fill_between(xaxis, 0, .4, where=xaxis < -1.62, alpha=.1, facecolor='slategrey')
    plt.annotate("Rechazo \nHipótesis Nula", xy=(1.96, .20))
    plt.annotate("Rechazo \nHipótesis Nula", xy=(-2.7, .20))
    plt.legend(loc = 8, fontsize=12)
    plt.ylim(0, .40)
    plt.title(r'Regiones de rechazo en la distribución de la nula $H_{0}\sim\mathcal{N}(0,1)$')
    plt.ylabel('Densidad')
    plt.xlabel('Rango')

def graph_significance():
    for i, p_value in enumerate([0.005, 0.025, 0.05]):
        significance_threshold(p_value, colors[i])

def gelman_hill_sim():
    """docstring for gelman_hill_sim"""
    birth_type = np.random.choice(['Fraternal', 'Identical', 'Single'], size=400, p=[1/125, 1/300, (1- 1/125 - 1/300)], replace=True)
    girls = np.full(400, 'NaN')
    for i in len(401):
        if birth_type[i] == "Single":
            girls[i] = np.random.binomial(1, .488, 1)
        elif birth_type[i] == "Identical":
            girls[i] = np.random.binomial(1, .495, 1)
        elif birth_type[i] == 'Fraternal':
            girls[i] = np.random.binomial(1, .495, 2)

def t_distribution(degree_freedom = [1, 5, 10, 30, 60]):
    """docstring for t_"""
    x_axis = np.linspace(-3, 3, 100)

    for i, degree in enumerate(degree_freedom):
        plt.plot(x_axis, stats.t.pdf(x_axis, degree), color=colors[i],
                 linestyle = '--', lw=2, label="Grados de Libertad: {}". format(degree))

    plt.plot(x_axis, stats.norm.pdf(x_axis), color=colors[5], label = r'$X\sim\mathcal{N}(0,1)$', lw=4)
    plt.legend()





def confidence_intervals():

    """docstring for confidence_intervals"""

    sims = 100
    coverage = np.empty(sims)
    lower_bound = np.empty(sims)
    upper_bound = np.empty(sims)

    for i in range(sims):
        draws = np.random.normal(loc=0, scale=math.sqrt(1), size=500)
        lower_bound[i] = draws.mean() - (draws.std() / math.sqrt(500)) * 1.96
        upper_bound[i] = draws.mean() + (draws.std() / math.sqrt(500)) * 1.96
        if (lower_bound[i] < 0) and (upper_bound[i] > 0):
            coverage[i] = 1
        else:
            coverage[i] = 0


    coverage_range = pd.DataFrame({
        'counter': list(range(1, sims + 1, 1)),
        'lb': lower_bound,
        'ub': upper_bound,
        'rejected': coverage
    })

    plt.axhline(y = 0, lw=3, color='#1c6cab')
    plt.annotate(r'$\theta$', xy=(101, .01), fontsize=20,color='#1c6cab')

    for i, row in coverage_range.iterrows():
        if row['rejected'] == 1:
            plt.vlines(row['counter'], row['lb'], row['ub'], color='dodgerblue')
        else:
            plt.vlines(row['counter'], row['lb'], row['ub'], color='tomato')
    
    plt.xlabel('Iteraciones')
    plt.ylabel('Parámetro')
    plt.title('')
