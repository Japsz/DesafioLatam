# Definición básica del conjunto de columnas definido en el desafío 1.
columns = [ 
    {'name': 'undp_hdi', 'isDiscrete': False},
    {'name': 'ccodealp', 'isDiscrete': True},
    {'name': 'ht_region', 'isDiscrete': True},
    {'name': 'ht_region_name', 'isDiscrete': True},
    {'name': 'gle_cgdpc', 'isDiscrete': False},
    {'name': 'imf_pop', 'isDiscrete': False},
    {'name': 'ffp_hf', 'isDiscrete': False},
    {'name': 'wef_qes', 'isDiscrete': False},
    {'name': 'wdi_expedu', 'isDiscrete': False},
    {'name': 'wdi_ners', 'isDiscrete': False}
]
# Función para mostrar el análisis del subset de columnas
def parseVariable(sample):
    for col in columns:
        if col['isDiscrete']:
            col['dsc'] = sample[col['name']].value_counts()
        else:
            col['dsc'] = sample[col['name']].describe()
            if col['name'] in ['gle_cgdpc', 'undp_hdi', 'imf_pop']:
                print('Variable Continua:')
                print(col['dsc'])
    return columns
# Función que analiza la pérdida de datos para una columna específica. print_list retorna las filas sin info
def lostData(dataframe, var, print_list = False):
    lostRows = dataframe[dataframe[var].isna()]
    numLost,_ = lostRows.shape
    numTotal,_ = dataframe.shape
    if print_list:
        return lostRows
    return {'count':numLost, 'percent': (numLost / numTotal)*100}
# Función que hace un histograma de una variable específica.
def makeHistPlot(dataframe, var, true_mean, sample_mean = False):
    plt.hist(dataframe[var])
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    if true_mean:
        plt.axvline(df[var].mean(), color="tomato")
    if sample_mean:
        plt.axvline(dataframe[var].mean(), color="green")
# Función que hacer un grafico de puntos para entre dos columnas.
def makeDotPlot(dataframe, plot_var, plot_by, global_state = False, statistic = 'mean'):
    plt.plot(dataframe[plot_var], dataframe[plot_by], marker='o', linestyle='none')
    plt.xlabel(plot_var)
    plt.ylabel(plot_by)
    if global_state:
        if statistic == 'mean':
            plt.axvline(df[plot_var].mean(), color="red")
        elif statistic == 'median':
            plt.axvline(df[plot_var].median(), color="orange")