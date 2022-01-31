from ds_charts import bar_chart, get_variable_types
from pandas import read_csv, DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show

register_matplotlib_converters()

filepath = '../../data/secondDataset/air_quality_tabular.csv'
data = read_csv(filepath, na_values='na', index_col='FID')

# Numero de colunas e linhas
print("Numero de linhas e colunas", data.shape)

# Cria um grafico com o numero de records (linhas) e varibles (colunas) 
figure(figsize=(8, 8))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('../images/dataDimensionality/air_quality_tabular_records_variables.png')
show()

# Troca o tipo de variaves simbolicas para category
cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
data.dtypes

# Imprime tipo por variaveis
print(data.dtypes)

# Cria grafico com o numero de variaveis por tipo binario, simbolico, data e numerico
variable_types = get_variable_types(data)
print(variable_types)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(8, 8))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('../images/dataDimensionality/air_quality_tabular_variable_types.png')
show()

# conta o numero de missing variables por variavel
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

# Cria grafico com o numero de missing variables por variavel
figure(figsize=(8, 8))
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
          xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('../images/dataDimensionality/air_quality_tabular_mv.png')
show()
