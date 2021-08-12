#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.stats as stats
from scipy.stats import binom, norm, t
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


colors = ['tomato', 'darkgoldenrod', 'limegreen', 'dodgerblue', 'sienna', 'slategray']


def bernoulli(p):
    """docstring for bernoulli"""
    return 1 if np.random.random() < p else 0

def binomial(n, p):
    """docstring for binomial"""
    return sum(bernoulli(p) for _ in range(n))

def binomial_sims():
    """docstring for binomial_sims"""
    xaxis = range(100)
    prob_range = [0.1, 0.2, 0.5, 0.7]
    colors = ['tomato', 'darkgoldenrod', 'limegreen', 'dodgerblue']
    for i, prob in enumerate(prob_range):
        plt.subplot(2, 2, i+1)
        plt.vlines(xaxis, 0, binom(100, prob).pmf(xaxis), color=colors[i])
        plt.title('X ~ Bin(100, {})'.format(prob))

def fdp_normal(x, mu = 0, sigma = 1):
    """docstring for fdp_normal"""
    bracket_exponencial = np.exp(-(x - mu) ** 2 / (2 * sigma **2))
    frac = np.sqrt(2 * np.pi) * sigma
    return (frac ** -1) * bracket_exponencial

def fdc_normal(x, mu=0, sigma=1):
    """docstring for fdc_normal"""
    elemental = 1 + math.erf((x - mu) / np.sqrt(2) / sigma)
    return elemental / 2

def normal_distributions():
    """docstring for normal_dsitributions"""
    # generamos in plano entre -3 y 3 con 100 observaciones
    xaxis = np.linspace(-3, 3, num = 100)

    # el primer gráfico
    plt.subplot(1, 2, 1)
    # genera la densidad x~n(0,1)
    plt.plot(xaxis, [fdp_normal(x) for x in xaxis], label='Normal Estandarizada')
    # genera la densidad de x~n(0,.5)
    plt.plot(xaxis, [fdp_normal(x, sigma=.5) for x in xaxis], label='Menor Varianza')
    # genera la densidad de x~n(-1, 1)
    plt.plot(xaxis, [fdp_normal(x, mu = -1) for x in xaxis], label='Menor Media')
    plt.title("Función de Densidad")
    plt.ylabel('Densidad')
    plt.xlabel("Rango")
    plt.legend()

    # el segundo gráfico
    plt.subplot(1, 2, 2)
    # genera la f(x)~n(0,1)
    plt.plot(xaxis, [fdc_normal(x) for x in xaxis], label='Normal Estandarizada', linestyle='--')
    # genera la f(x)~n(0,.5)
    plt.plot(xaxis, [fdc_normal(x, sigma=.5) for x in xaxis], label='Menor Varianza', linestyle='--')
    # genera la f(x)~n(-1, 1)
    plt.plot(xaxis, [fdc_normal(x, mu=-1) for x in xaxis], label='Menor Media', linestyle='--')
    plt.title("Función Acumulada")
    plt.ylabel("Densidad")
    plt.xlabel("Rango")
    plt.legend()

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

def plot_hist(p, n, points):
    # genera un array temporal para guardar distribuciones binomiales repetidas `points` veces
    tmp = [binomial(n, p) for _ in range(points)]
    # contador de instancias
    hist = Counter(tmp)
    # delimitador de ancho de columnas en histograma
    bins = [x -0.4 for x in hist.keys()]
    # estimador de densidad
    density = [v / points for v in hist.values()]
    # plot densidad a lo largo de bins
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
    #para cada indice y valor en los valores listados
    for i, v in enumerate(values):
        # generamos un subplot entre 1 y 6
        plt.subplot(2, 3, i+1)
        # insertamos el valor
        plot_hist(.2, 1000, v)

def confidence_intervals():
    # definimos cantidad de simulaciones
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

    # iniciar contador
    cnt = []
    # para cada indice entre pares de intervalos
    for i, ci in enumerate(zip(lower_bound, upper_bound)):
        cnt.append(i+1)

    # guardemos la información en un dataframe para facilitar la manipulación
    coverage_range = pd.DataFrame({'counter': cnt,
                                   'lb': lower_bound,
                                   'ub': upper_bound,
                                   'rejected' : coverage})
    # demarcar parámetro poblacional
    plt.axhline(y=0, lw=3, color='#1c6cab')
    plt.annotate(r'$\theta$ ', xy=(101,.01), fontsize=20, color='#1c6cab')

    for i, row in coverage_range.iterrows():
        if row['rejected'] == 1:
            plt.vlines(row['counter'], row['lb'], row['ub'], color='dodgerblue', linewidth=1.5)
        else:
            plt.vlines(row['counter'], row['lb'], row['ub'], color='tomato', linewidth=3)

    plt.xlabel('Iteraciones')
    plt.ylabel('Parámetro')
    plt.title('')

def significance_threshold(cutoff, c):
    xaxis = np.linspace(-3, 3, 500)
    t_distribution = stats.t.pdf(xaxis, 500)
    cutoff_point = stats.t.ppf(cutoff, 500)
    plt.plot(xaxis, t_distribution, color='#1c6cab', lw=3)
    plt.axvline(cutoff_point, 0, 0.4, color=colors[i],
                label=r'Sig: {0}% z: {1}'.format(int((1-(2*cutoff)) * 100), round(cutoff_point, 2)),
                linestyle='--')
    plt.annotate("{}".format(-cutoff), xy=(cutoff_point-.25, 0.16), color=colors[i], )
    plt.axvline(-cutoff_point, 0, 0.4, color=colors[i], linestyle='--')
    plt.annotate("{}".format(cutoff), xy=(-cutoff_point, 0.16), color=colors[i])
    plt.annotate("Falla en Rechazar \nHipótesis Nula", xy=(-0.35, .25))
    plt.fill_between(xaxis, 0, .4, where=xaxis > 1.62, alpha=.1, facecolor='slategrey')
    plt.fill_between(xaxis, 0, .4, where=xaxis < -1.62, alpha=.1, facecolor='slategrey')
    plt.annotate("Rechazo \nHipótesis Nula", xy=(1.96, .20))
    plt.annotate("Rechazo \nHipótesis Nula", xy=(-2.7, .20))
    plt.legend(loc=8, fontsize=12)
    plt.ylim(0, .40)
    plt.title(r'Regiones de rechazo en la distribución de la nula $H_{0}\sim\mathcal{N}(0,1)$')
    plt.ylabel('Densidad')
    plt.xlabel('Rango')

def graph_significance():
    for i, p_value in enumerate([0.005, 0.025, 0.05, 0.10]):
        significance_threshold(p_value, colors[i])

def histogram_gaussian(mu, sigma, n_size):
    """docstring for histogram_gaussian"""
    plt.hist(np.random.normal(mu, sigma, n_size), color='grey', alpha=.4, normed=True)
    x_min, x_max = plt.xlim()
    plt.plot(np.linspace(x_min, x_max, n_size),
             stats.norm.pdf(np.linspace(x_min, x_max, n_size), mu, sigma),
             color='tomato', lw=3)
    plt.axvline(mu, color='dodgerblue', linestyle='--', lw=3)

def normal_distribution_sigma():
    """docstring for normal_distribution_sigma"""
    xaxis = np.linspace(-4, 4, 1000)
    x_axis_sigma = np.linspace(-1, 1, 1000)
    x_axis_sigma_2 = np.linspace(-2, 2, 1000)
    x_axis_sigma_3 = np.linspace(-3, 3, 1000)
    plt.plot(xaxis, stats.norm.pdf(xaxis, 0, 1), color='grey', alpha=.8, lw=1)
    plt.fill_between(x_axis_sigma, stats.norm.pdf(x_axis_sigma, 0, 1),
                     color='dodgerblue', alpha=.9, label=r'$\pm 1 \sigma$: 68%')
    plt.fill_between(x_axis_sigma_2, stats.norm.pdf(x_axis_sigma_2, 0, 1),
                     color='dodgerblue', alpha=.6, label=r'$\pm 2 \sigma$: 95%')
    plt.fill_between(x_axis_sigma_3, stats.norm.pdf(x_axis_sigma_3, 0, 1),
                     color='dodgerblue', alpha=.2, label=r'$\pm 3 \sigma$: 99%')
    plt.xlabel(r'Desviación Estandar $\sigma$')
    plt.title(r'Distribución Normal Estandarizada: $X\sim \mathcal{N}(0, 1)$')
    plt.legend()
