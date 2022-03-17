# Transformamos las fechas de cada semana a pd.Int64Index
def convertWeeklyIndex(df):
    cols = [*df.columns, 'month', 'year', 'weekOfYear']
    intDf = pd.DataFrame(columns=cols)
    # Desde 2019-13 a 2021-48 hay 134 semanas
    año = 2020
    semana = 9
    index = 1
    debugLabels = []
    while True:
        cero = '0' if semana < 10 else ''
        label = f'{año}-{cero}{semana}'
        fecha = pd.to_datetime(f'{label}-6', format='%Y-%U-%w')
        month = fecha.month
        year = fecha.year
        weekOfYear = fecha.strftime('%U')
        if label in df.index:
            intDf.at[index] = [*df.loc[label, :], *[month, year, weekOfYear]]
            debugLabels.append(label)
        else:
            print('weekly No encontrado', label, index)
            intDf.at[index] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, month, year, weekOfYear]
        if label == '2022-09':
            break
        index += 1
        semana += 1
        if semana > 52:
            año += 1
            if año == 2020:
                semana = 0
            else:
                semana = 1
    # Creamos un nuevo DataFrame con las fechas de las semanas
    newIndex = pd.RangeIndex(0, index)
    try :
        intDf.index = newIndex
    except:
        # Buscar el index en df que no esta en debugLabels
        notFound = list(filter(lambda x: x not in debugLabels, df.index))
        print(notFound)
    return intDf

# Transformamos las fechas de cada semana a pd.Int64Index
def convertMonthlyIndex(df):
    cols = [*df.columns, 'month', 'year']
    intDf = pd.DataFrame(columns=cols)
    # Desde 2019-13 a 2021-48 hay 134 semanas
    año = 2020
    mes = 3
    index = 1
    indexes = []
    debugLabels = []
    while True:
        cero = '0' if mes < 10 else ''
        label = f'{año}-{cero}{mes}'
        if label in df.index:
            intDf.at[index] = [*df.loc[label, :], *[mes, año]]
            debugLabels.append(label)
        else:
            print('monthly No encontrado', label, index)
            intDf.at[index] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, mes, año]
        if label == '2022-02':
            break
        index += 1
        mes += 1
        if mes > 12:
            mes = 1
            año += 1
    # Creamos un nuevo DataFrame con las fechas de las semanas
    newIndex = pd.RangeIndex(0, index)
    try :
        intDf.index = newIndex
    except:
        # Buscar el index en df que no esta en debugLabels
        notFound = list(filter(lambda x: x not in debugLabels, df.index))
        print(notFound)
    return intDf

def preprocessSeries(subDf, freq = 'D', cols = ['kilos_sum', 'kilos_std', 'kilos_max', 'kilos_median']):
    filler = MissingValuesFiller(fill=0.)
    #Instanciamos
    itemTimeSeries = TimeSeries.from_dataframe(subDf, None, value_cols=cols, freq=freq)
    # Completamos valores faltantes
    itemTimeSeries = filler.transform(itemTimeSeries)
    # Separamos los datos en train y test
    train, test = itemTimeSeries.split_before(0.8)
    # Separamos las covariantes
    scaled_train_covs = concatenate(
        [
            train['kilos_max'],
            train['kilos_std'],
            train['kilos_median'],
        ],
        axis="component",
    )
    scaled_test_covs = concatenate(
        [
            test['kilos_max'],
            test['kilos_std'],
            test['kilos_median'],
        ],
        axis="component",
    )
    return {
        'df': subDf,
        'series': itemTimeSeries,
        'train': {
            'target': train['kilos_sum'],
            'past_covars': scaled_train_covs,
        },
        'test': {
            'target': test['kilos_sum'],
            'past_covars': scaled_test_covs,
        }
    }