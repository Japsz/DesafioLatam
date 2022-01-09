#!/usr/bin/python3

"""
simula una cantidad n de observaciones que siguen una especificación:
age - Un número entero aleatorio entre 18 y 90
income - Un número entero aleatorio entre 10000 y 1000000.
employment_status - Un string entre Employed con probabilidad de ocurrencia .7
y Unemployed con probabilidad de ocurrencia .3.
debt_status - Un string entre Debt con probabilidad de ocurrencia .2 y No Debt
con una probabilidad de ocurrencia de .8
churn_pr - Probabilidad predicha de churn para el usuario siguiendo una
distribución BetaBinomial(alpha=600, beta=300).
uso:

python3.6 script_demo.py filename seed n_rows

Donde:
* filename hace referencia a al titulo del archivo csv de salida.
* seed es la semilla pseudoaleatoria a utilizar.
* n_rows es la cantidad de filas a generar.
"""

import random
import numpy as np
import csv
import sys


def create_random_row():
	deliverer_id = np.random.choice(range(100), 1)[0]
	delivery_zone = np.random.choice(['I', 'II', 'III', 'IV', 'V', 'VI', 'VIII'])
	monthly_apply_usage = np.random.poisson(15)
	subscription_type = np.random.choice(['Free' 'Prepaid', 'Monthly', 'Trimestral', 'Semestral', 'Yearly'], 
											1, 
											[.3, .2, .1, .15, .2, .05])[0]
	paid_price = np.random.normal(25.45, 10)
	customer_size = np.random.poisson(2)+1
	menu = np.random.choice(['Asian', 'Indian', 'Italian', 'Japanese', 'French', 'Mexican'], 1)[0]
	delay_time = np.random.normal(10, 3.2)

	return [deliverer_id, delivery_zone, monthly_apply_usage, subscription_type, paid_price, customer_size, menu, delay_time]

filename = sys.argv[1]
seed_number = int(sys.argv[2])
n_rows = int(sys.argv[3])

np.random.seed(seed_number)

with open(f"{filename}.csv", 'w') as csvfile:
	file = csv.writer(csvfile, delimiter=',')#, quotechar='|', quoting=csv.QUOTE_MINIMAL)
	file.writerow(['deliverer_id', 'delivery_zone', 'monthly_apply_range', 'subscripcion_type', 'paid_price', 'customer_size',
			'menu', 'delay_time'])
	for _ in range(n_rows):
		file.writerow(create_random_row())
print("Script Listo!")