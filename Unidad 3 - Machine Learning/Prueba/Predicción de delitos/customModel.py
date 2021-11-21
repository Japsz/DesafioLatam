# importamos el hash de labels de nuestro archivo helper creado en base al archivo xlsx
import refactoringParams as rfParams2
import pandas as pd
import numpy as np
import re
import datetime
# Creamos nuestra clase customizada para nuestro modelo
class CustomModel:
    def __init__(self, model, modelScaler, hasMixture, gaussianMixture, gaussianScaler, target):
        self.model = model
        self.gaussianMixture = gaussianMixture
        self.hasMixture = hasMixture
        self.gaussianScaler = gaussianScaler
        self.target = target
        self.modelScaler = modelScaler
    def predict(self, X):
        refactoredDf = self.transform(X)
        return self.model.predict(refactoredDf)

    def predict_proba(self, X):
        refactoredDf = self.transform(X)
        return self.model.predict_proba(refactoredDf)
    def preprocess(self, tweetSeries):
        copy = pd.DataFrame([])
        # Agregamos las columnas
        copy['n_words'] = tweetSeries.apply(lambda x: len(x.split()))
        copy['n_chars'] = tweetSeries.apply(lambda x: len(x))
        # Conseguir la cantidad de menciones y hashtags
        copy['n_mentions'] = tweetSeries.apply(lambda x: len(re.findall(r'@\w+', x)))
        copy['n_hashtags'] = tweetSeries.apply(lambda x: len(re.findall(r'#\w+', x)))
        # Reemplazar las menciones y hashtags por una palabra
        copy['content'] = tweetSeries.apply(lambda x: re.sub(r'@\w+', '@@@', x))
        copy['content'] = copy['content'].apply(lambda x: re.sub(r'#\w+', '###', x))
        return copy
# Funcion que refactoriza el dataframe de la base de datos
    def transform(self, df):
        # Quitamos los atributos que no nos interesan
        refactoredDf = df.drop(columns=rfParams2.removeLabels)
        if 'Unnamed: 0' in df.columns:
            refactoredDf = refactoredDf.drop(columns=['Unnamed: 0'])

        # Se inicializan las series para los atributos a agregar
        imcSerie = pd.Series(dtype='float64')
        dateSerie = pd.Series(dtype='object')
        monthSerie = pd.Series(dtype='object')
        daytimeSerie = pd.Series(dtype='object')
        # Se itera sobre las filas del df original
        for i, row in df.iterrows():
            # Se agrega el imc en kg/m2 (considerando weight en lb y height en ft)
            imcSerie.at[i] = (row['weight']*.453)/((row['ht_feet']*12*.0254) + (row['ht_inch']*.0254))**2
            # Se parsea la fecha y hora del parametro en un string ISO
            strdate = str(row['datestop'])
            strTime = str(row['timestop'])
            year = strdate[-4:]
            date = strdate[-6:-4]
            month = strdate[0:-6]
            hour= strTime[-4:-2]
            minute = strTime[-2:]
            if hour == '' or int(hour) >= 24:
                hour = '00'
            if len(hour) == 1:
                hour = '0' + hour
            if len(minute) == 1:
                minute = '0' + minute
            if len(month) == 1:
                month = '0' + month
            # Se consigue la instancia de fecha considerando la hora en NYC
            isoString = f'{year}-{month}-{date}T{hour}:{minute}:00-04:00'
            dateStop = datetime.datetime.fromisoformat(isoString)
            # Se agrega la info a la serie con información de dia y mes
            dateSerie.at[i] = dateStop.strftime('%a')
            monthSerie.at[i] = dateStop.strftime('%b')
            # Se agrega el valor del daytime
            if int(hour) >= 6 and int(hour) <= 11:
                daytimeSerie.at[i] = 'morning'
            elif int(hour) >= 12 and int(hour) <= 17:
                daytimeSerie.at[i] = 'afternoon'
            elif int(hour) >= 18 and int(hour) <= 23:
                daytimeSerie.at[i] = 'night'
            else:
                daytimeSerie.at[i] = 'early_morning'
        # Se agrega la serie al dataframe
        refactoredDf = pd.concat([
            refactoredDf,
            pd.get_dummies(dateSerie, prefix='day'),
            pd.get_dummies(monthSerie, prefix='month'),
            pd.get_dummies(daytimeSerie, prefix='daytime')
        ], axis=1)
        refactoredDf['imc'] = imcSerie

        # Se hace one-hot encoding de los atributos categoricos multiclase
        for col in rfParams2.multiClassLabels:
            refactoredDf = pd.concat([refactoredDf, pd.get_dummies(refactoredDf[col], prefix=col)], axis=1)
            refactoredDf = refactoredDf.drop(columns=col)
        # Se hace one-hot encoding de los atributos categoricos binarios
        for col in rfParams2.binaryLabels:
            refactoredDf = pd.concat([refactoredDf, pd.get_dummies(refactoredDf[col], prefix=col, drop_first=True)], axis=1)
            refactoredDf = refactoredDf.drop(columns=col)
        # Nos aseguramos de convertir las variables numéricas a float
        for col in rfParams2.numericLabels:
            refactoredDf[col] = refactoredDf[col].astype('float64')
        # Se quita el vector objetivo
        if self.target == 'arst':
            refactoredDf = refactoredDf.drop(columns=['arstmade_Y'])
        else:
            violentColumns = ['pf_hands', 'pf_wall', 'pf_grnd', 'pf_drwep', 'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_pepsp', 'pf_other']
            refactoredDf = refactoredDf.drop(columns=[f'{x}_Y' for x in violentColumns])
        # Se agrega la GMM si lo necesita
        if self.hasMixture:
            X = pd.concat([
                df.loc[:, ['repcmd', 'revcmd']],
                pd.get_dummies(df.loc[:, ['pct', 'addrpct', 'detailcm']]),
                pd.get_dummies(df['sumoffen']),
                pd.get_dummies(df['crimsusp']),
            ], axis=1)
            # Guardamos el dataframe con las probabilidades de cada cluster
            gmmProbaDf = pd.DataFrame(self.gaussianMixture.predict_proba(self.gaussianScaler.transform(X)), columns=['GMM_' + str(x) for x in range(9)])
            return pd.concat([refactoredDf, gmmProbaDf], axis=1)
        return self.modelScaler.transform(refactoredDf)
