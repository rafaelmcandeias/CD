from ds_charts import HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, choose_grid, HEIGHT, bar_chart
from pandas import read_csv
from matplotlib.pyplot import figure, savefig, show

filepath = '../../data/firstDataset/NYC_collisions_tabular.csv'
data = read_csv(filepath)

numeric = get_variable_types(data)['Numeric']
symbolic = get_variable_types(data)['Symbolic']
date = get_variable_types(data)['Date']

print(numeric)

# Cria um grafico com o numero de records (linhas) e varibles (colunas) 
figure(figsize=(8, 8))
counts = {}
for tp in numeric:
    counts[tp] = len(numeric[tp])
figure(figsize=(8, 8))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('../images/dataGranularity/NYC_collisions_numeric_variables.png')

figure(figsize=(8, 8))
counts = {}
for tp in symbolic:
    counts[tp] = len(symbolic[tp])
bar_chart(list(counts.keys()), list(counts.values()), title='Numeric Variables')
savefig('../images/dataGranularity/NYC_collisions_symbolic_variables.png')

figure(figsize=(8, 8))
counts = {}
for tp in date:
    counts[tp] = len(date[tp])
bar_chart(list(counts.keys()), list(counts.values()), title='Numeric Variables')
savefig('../images/dataGranularity/NYC_collisions_date_variables.png')

filepath = '../../data/secondDataset/air_quality_tabular.csv'
data = read_csv(filepath)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}



# Devolve um  erro se nao houver variaveis numericas
variables = get_variable_types(data)['Numeric']
symbolic_vars = get_variable_types(data)['Symbolic']
binary_vars = get_variable_types(data)['Binary']



if [] == variables:
    raise ValueError('There are no numeric variables.')


# Cria um histograma para cada variavel numeric
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
