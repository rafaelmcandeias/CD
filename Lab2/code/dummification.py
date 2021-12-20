from numpy import number
from sklearn.preprocessing import OneHotEncoder
from ds_charts import get_variable_types
from pandas import DataFrame, concat, read_csv
from pandas.plotting import register_matplotlib_converters


def dummify(df, vars_to_dummify):
    print(vars_to_dummify)
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    x = df[vars_to_dummify]
    print(x)
    encoder.fit(x)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(x)
    dummy = DataFrame(trans_X, columns=new_vars, index=x.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)
    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df


register_matplotlib_converters()
filename = '../../data/secondDataset/air_quality_tabular.csv'
data = read_csv(filename, index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)

# Drop out all records with missing values
data.dropna(inplace=True)

# dummifies all symbolic variables
variables = get_variable_types(data)
symbolic_vars = variables['Symbolic']
df = dummify(data, symbolic_vars)
df.to_csv(f'../data/air_quality_tabular_dummified.csv', index=False)

df.describe(include=[bool])