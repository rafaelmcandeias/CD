# NYC COllISIONS DATA SET

from pandas import read_csv, unique
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

file = 'NYC_COLLISIONS'
filename = 'firstDataSet/NYC_collisions_tabular.csv'
df = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True, infer_datetime_format=True)

# print(get_variable_types(df))

print("----------------------DV------------------------")
# counts the number of different values in each column(ids have a very large number)
for c in df.columns:
    print(c, len(df[c].dropna(inplace=False).unique()))

print("----------------------MV------------------------")

for c in df.columns:
    print(c, df[c].isna().sum())


