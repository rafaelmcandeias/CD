from ds_charts import HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, choose_grid, HEIGHT
from pandas import read_csv

filepath = '../../data/secondDataset/air_quality_tabular.csv'
data = read_csv(filepath)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}


# Devolve um  erro se nao houver variaveis numericas
variables = get_variable_types(data)['Numeric']
if [] == variables:
    raise ValueError('There are no numeric variables.')


# Cria um histograma para cada variavel
rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s' % variables[n])
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins=100)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('../images/dataGranularity/seconddataset_granularity_single.png')
show()


# Cria um histograma com diferentes bins (numero de barras) para as variaveis com os histogramas mais uniformes
variable = 'Field_1' # para a firstdataset usei o COLLISION_ID
bins = (10, 100, 1000, 10000)
fig, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for %s %d bins' % (variable, bins[j]))
    axs[j].set_xlabel(variable)
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data[variable].values, bins=bins[j])
savefig(f'../images/dataGranularity/seconddataset_granularity_study_{variable}.png')
show()


# Cria um histograma com bins 10 100 e 1000 para todas as vars numericas
rows = len(variables)
bins = (10, 100, 1000)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins' % (variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
savefig('../images/dataGranularity/seconddataset_granularity_study.png')
show()
