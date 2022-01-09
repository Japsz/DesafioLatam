#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    return df.drop(column_name, axis=1)

df = pd.read_csv('train_delivery_data.csv')
# Se recodifica variable objetivo para aquellos con valor mayor a la media 
delayMean = df['delay_time'].mean()
df['delay_time'] = np.where(df['delay_time'] > delayMean, 1, 0)
# Se refactorizan las variables categóricas
df = create_dummies(df, 'delivery_zone')
df = create_dummies(df, 'subscription_type')
df = create_dummies(df, 'menu')
# Se crea una variable de entrada para el modelo
X = df.drop(['delay_time'], axis=1)
# Se crea una variable de salida para el modelo
y = df['delay_time']
# Se separa el conjunto de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11238)
# Se generan y entrenan los modelos
models = [
    LogisticRegression(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    BernoulliNB(),
    DecisionTreeClassifier()
]
for model in models:
    model.fit(X_train, y_train)
    # Se guardan los modelos entrenados
    joblib.dump(model, 'model_{}.pkl'.format(model.__class__.__name__))
    # Se evaluan los modelos
    y_pred = model.predict(X_test)
    print(model.__class__.__name__)
    print(classification_report(y_test, y_pred))
    print('='*50)

