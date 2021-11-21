import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Creamos nuestra clase customizada para nuestro modelo
class CustomModel:
    def __init__(self, model, tfidfLatentVectorizer, ldaModel, tfidfVectorizer):
        self.model = model
        self.ldaModel = ldaModel
        self.tfidfVectorizer = tfidfVectorizer
        self.tfidfLatentVectorizer = tfidfLatentVectorizer

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
    def transform(self, X):
        df = self.preprocess(X)
        # Conseguimos la matriz TF-IDF de las columnas apartadas
        testLatentMatrix = self.tfidfLatentVectorizer.transform(df['content'])
        # Generamos el dataframe con las probabilidades de tópico por documento y lo concatenamos al dataframe original
        topics_for_each_doc_test = pd.DataFrame(np.round(self.ldaModel.transform(testLatentMatrix), 3), index=df.index )
        topics_for_each_doc_test.columns = list(map(lambda x: "T: {}".format(x), range(1, self.ldaModel.n_components + 1)))
        partialDf = pd.concat([df.drop(columns=['content']), topics_for_each_doc_test], axis=1)
        # Generamos la transformación TF-IDF de las columnas conservadas
        tfidfMatrix = self.tfidfVectorizer.transform(df['content'])
        # Retornamos el df final con toda la información
        return pd.concat([partialDf, pd.DataFrame(tfidfMatrix.toarray(), columns=self.tfidfVectorizer.get_feature_names())], axis=1)