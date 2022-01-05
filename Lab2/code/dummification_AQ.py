from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame, concat, read_csv
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
file = "air_quality"
filename = '../../data/secondDataset/air_quality_tabular_no_mvs.csv'
data = read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)

# Drop out all records with missing values
# should be 0
data.dropna(inplace=True)

def dummify_Label(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    final_df = df[other_vars]

    for var in vars_to_dummify:
        encoder = LabelEncoder()
        encoder.fit(df[var])
        trans = encoder.transform(df[var])
        dummy = DataFrame(trans, columns=[var], index=df[var].index)
        dummy = dummy.convert_dtypes(convert_boolean=True)
        final_df = concat([final_df, dummy], axis=1)

    return final_df

df = dummify_Label(data, ['ALARM'])
df.to_csv(f'../../data/secondDataset/air_quality_tabular_dummified.csv', index=False)
df.describe(include=[bool])
