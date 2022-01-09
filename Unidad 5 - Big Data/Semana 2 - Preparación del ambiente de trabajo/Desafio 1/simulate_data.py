#!/usr/bin/python
# Benjamín Meneses
from os import linesep
import pandas as pd
import numpy as np
import csv
import sys
# Crear csv con datos simulados
def create_csv(filename, n_samples, random_seed):
    np.random.seed(random_seed)
    with open(filename, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile)
        csv_columns = ['deliverer_id', 'delivery_zone', 'monthly_app_usage', 'subscription_type', 'paid_price', 'customer_size', 'menu', 'delay_time']
        csvwriter.writerow(csv_columns)
        for i in range(n_samples):
            deliverer_id = np.random.choice(range(100), 1)[0]
            delivery_zone = np.random.choice(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'])
            monthly_app_usage =  np.random.poisson(15)
            subscription_type =  np.random.choice(['Free','Prepaid','Monthly', 'Trimestral', 'Semestral', 'Yearly'], 1,[.30, .20, 10, .15, .20, .05])[0]
            paid_price =  np.random.normal(25.45, 10)
            customer_size =  np.random.poisson(2) + 1
            menu =  np.random.choice(['Asian', 'Indian', 'Italian', 'Japanese','French', 'Mexican'],1)[0]
            delay_time =  np.random.normal(10,3.2)
            csvwriter.writerow([deliverer_id, delivery_zone, monthly_app_usage, subscription_type, paid_price, customer_size, menu, delay_time])
    
create_csv(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))



