import pandas as pd

df = pd.read_csv('./athlete_events.csv')

ejercicio_1 = df.shape
# Se asume que el enunciado donde dice 'Competencias' se refiere a la columna 'Event' en vez de 'Games' que se podria asumir como el conjunto de competencias
ejercicio_2, = df['Event'].value_counts().shape

biSeasonalAthletes = float(df.groupby('ID').filter(lambda x: x['Season'].nunique() > 1)['ID'].nunique())
ejercicio_3 = biSeasonalAthletes / len(df.groupby('ID'))

ejercicio_4 = df[df['Season'] == 'Summer'].loc[:, ['Year', 'City']].min()['City']
ejercicio_5 = df.loc[:, ['Year', 'City']][df['Season'] == 'Winter'].min()['City']
ejercicio_6 = list(df.groupby('NOC').nunique().sort_values('ID').tail(10).index)

gold = df[df['Medal'] != 'NA'][df['Medal'] == 'Gold'].value_counts().shape[0]
silver = df[df['Medal'] != 'NA'][df['Medal'] == 'Silver'].value_counts().shape[0]
bronze = df[df['Medal'] != 'NA'][df['Medal'] == 'Bronze'].value_counts().shape[0]
total_medal = df[df['Medal'] != 'NA'].value_counts().shape[0]
ejercicio_7 = ( gold/total_medal, silver/total_medal, bronze/total_medal )

firstSummerGame =  df[df['Season'] == 'Summer'].loc[:, ['Year', 'Games']].min()['Games']
ejercicio_8 = df[df['Games'] == firstSummerGame]['NOC'].unique() 