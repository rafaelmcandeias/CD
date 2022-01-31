from ds_charts import HEIGHT, get_variable_types
from matplotlib.pyplot import savefig, show, subplots
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

# Start
register_matplotlib_converters()
filepath = '../../data/firstDataset/NYC_collisions_tabular.csv'
data = read_csv(filepath, index_col='UNIQUE_ID', parse_dates=True, infer_datetime_format=True)

# Da erro se nao existirem variaveis numericas
numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

# Cria um scatterplot para todas as variaveis numericas
rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    for j in range(i+1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'../images/dataSparsity/NYC_collisions_sparsity_study_numeric.png')
show()

# Da erro se nao houver variaveis simbolicas
symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

# Passa todas as variaveis simbolicas para string
for i in symbolic_vars:
    data[i] = data[i].astype(str)

# Cria um scatterplot para todas as variaveis simbolicas
rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(symbolic_vars)):
    var1 = symbolic_vars[i]
    for j in range(i+1, len(symbolic_vars)):
        var2 = symbolic_vars[j]
        axs[i, j-1].set_title("%s x %s" % (var1, var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
print("saving fig")
savefig(f'../images/dataSparsity/NYC_collisions_sparsity_study_symbolic.png')
show()
