import pandas as pd
from joblib import load
import sys
import numpy as np
"""
python predict_model.py filename_test best_model_filename
"""

filename_test = sys.argv[1]
best_model_filename = sys.argv[2]

df_test = pd.read_csv(filename_test)

df_test['delay_time_bin'] = np.where(df_test['delay_time'] > df_test['delay_time'].mean(), 1, 0) # Delay_time recoding
df_test.drop(columns = ['delay_time'], inplace = True)
#df_test['deliverer_id'] = df_test['deliverer_id'].astype('O')
df_test = pd.get_dummies(df_test, drop_first = True)

model = load(best_model_filename)

#file = open('eval_pr.txt', 'w')

df_test['delay_time_predicts'] = model.predict(df_test.drop(columns = ['delay_time_bin']))

# 1. Prob de atraso en pedido por sobre la media para cada zona de envio
#file.write('\n################## Prob. de atraso por mas de la media para cada zona ################## \n')
print('\n################## Prob. de atraso por mas de la media para cada Zona ################## \n')
for zone in df_test.filter(regex = 'delivery_zone', axis = 1).columns:
    prob = round(df_test[df_test[zone] == 1]['delay_time_predicts'].mean(), 2)
    print('Prob para {}: {}'.format(zone, prob))
#    file.write('Prob para {}: {}'.format(zone, prob))

# 2. Prob de atraso en pedido por sobre la media para cada repartidor
#file.write('\n################## Prob. de atraso por mas de la media para cada repartidor ################## \n')
print('\n################## Prob. de atraso por mas de la media para cada Repartidor ################## \n')
deliverer_grouped = df_test.groupby(by = 'deliverer_id')['delay_time_predicts'].mean()
print(deliverer_grouped)
#[print(f'Prob para {deliverer}: {prob}') for deliverer, prob in deliverer_grouped]
#[file.write(f'Prob para {deliverer}: {prob}') for deliverer, prob in deliverer_grouped]

#for col in df_test.filter(regex = 'deliverer_id', axis = 1).columns:
#    prob = round(df_test[df_test[col] == 1]['delay_time_predicts'].mean(), 2)
#    print('Prob para {}: {}'.format(col, prob))
#    file.write('Prob para {}: {}'.format(col, prob))


# 3. Prob de atraso en pedido por sobre la media para cada repartidor
#file.write('\n################## Prob. de atraso por mas de la media para cada Menu ################## \n')
print('\n################## Prob. de atraso por mas de la media para cada Menu ################## \n')
for col in df_test.filter(regex = 'menu', axis = 1).columns:
    prob = round(df_test[df_test[col] == 1]['delay_time_predicts'].mean(), 2)
    print('Prob para {}: {}'.format(col, prob))
#    file.write('Prob para {}: {}'.format(col, prob))


# 2. Prob de atraso en pedido por sobre la media para cada repartidor
#file.write('\n################## Prob. de atraso por mas de la media para cada Subscripcion ################## \n')
print('\n################## Prob. de atraso por mas de la media para cada Subscripcion ################## \n')
for col in df_test.filter(regex = 'subscripcion_type', axis = 1).columns:
    prob = round(df_test[df_test[col] == 1]['delay_time_predicts'].mean(), 2)
    print('Prob para {}: {}'.format(col, prob))
#    file.write('Prob para {}: {}'.format(col, prob))

#file.close()
