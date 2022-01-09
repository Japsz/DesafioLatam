#!/usr/bin/python
import pandas as pd
import numpy as np
import joblib

pd.options.display.max_rows = 999
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    return df.drop(column_name, axis=1)

pre_df = pd.read_csv('test_delivery_data.csv')

# Se recodifica variable objetivo para aquellos con valor mayor a la media
df = pre_df.copy()
delayMean = df['delay_time'].mean()
df['delay_time'] = np.where(df['delay_time'] > delayMean, 1, 0)
# Se refactorizan las variables categóricas
df = create_dummies(df, 'delivery_zone')
df = create_dummies(df, 'subscription_type')
df = create_dummies(df, 'menu')
# Se crea una variable de entrada para el modelo
X = df.drop(['delay_time'], axis=1)
# Se cargan los modelos
bestModel = joblib.load('model_RandomForestClassifier.pkl')
# Se cargan las predicciones
y_pred = bestModel.predict(X)
# Se agrega al preprocesado
pre_df['delay_time_pred'] = y_pred
# Se muestran las probabilidades pedidas:
# Probabilidad de delay por zona de entrega
print(pre_df.groupby('delivery_zone')['delay_time_pred'].mean())
# Probabilidad de delay por repartidor
print(pre_df.groupby('deliverer_id')['delay_time_pred'].mean())
# Probabilidad de delay por tipo de menú
print(pre_df.groupby('menu')['delay_time_pred'].mean())
# Probabilidad de delay por tipo de suscripción
print(pre_df.groupby('subscription_type')['delay_time_pred'].mean())
