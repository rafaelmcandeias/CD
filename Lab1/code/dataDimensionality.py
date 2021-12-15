from ds_charts import bar_chart
from matplotlib.pyplot import figure, savefig, show
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# filename = 'data/algae.csv'
# data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

filename = '../data/firstDataset/NYC_collisions_tabular.csv'
data = read_csv(filename, na_values='na')

# Numero de colunas e linhas
print(data.shape)

# Imprime um grafico com N de linhas e N de colunas
# figure(figsize=(4, 2))
# values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
# bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
# savefig('images/first_dataset_records_variables.png')
# show()

# Complexidade 
complexidade = data.shape[0] * data.shape[1]
print("Complexidade:", complexidade)

# Imprime tipo por variavels
print(data.dtypes)

# Missing values
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

total = 0
for v in mv.values():
    total += v
print("Missing values:", total, "%:", total/complexidade)

