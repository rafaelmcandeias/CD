from imblearn.over_sampling import SMOTE
from pandas import read_csv, concat, DataFrame, Series
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
from sklearn.model_selection import train_test_split

class_var = 'ALARM'

for scalling in ("minmax", "zscore"):
    data = read_csv(f'../../Lab3/data/scalling/air_quality_scaled_{scalling}.csv')
    X = data.drop(class_var, axis=1)
    y = data[class_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    data_train = concat([X_train, y_train], axis=1)
    data_train.to_csv(f'../../data/secondDataset/air_quality_{scalling}_train.csv', index=False)
    data_test = concat([X_test, y_test], axis=1)
    data_test.to_csv(f'../../data/secondDataset/air_quality_{scalling}_test.csv', index=False)

for scalling in ("minmax", "zscore"):
    original = read_csv(f'../../data/secondDataset/air_quality_{scalling}_train.csv', sep=',', decimal='.')
    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(
        target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class],
                        target_count[negative_class]]}

    figure()
    bar_chart(target_count.index, target_count.values, title='Class balance')
    savefig(f'../data/balancing/images/air_quality_{scalling}_train_balance.png')

    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]

    # UNDER SAMPLING
    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_under = concat([df_positives, df_neg_sample], axis=0)
    df_under.to_csv(f'../data/balancing/air_quality_{scalling}_train_undersampling.csv', index=False)
    values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
    print("under-sampling:")
    print('Minority class=', positive_class, ':', len(df_positives))
    print('Majority class=', negative_class, ':', len(df_neg_sample))
    print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')

    #OVER SAMPLING
    print("over-sampling:")
    df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives], axis=0)
    df_over.to_csv(f'../data/balancing/air_quality_{scalling}_train_oversampling.csv', index=False)
    values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
    print('Minority class=', positive_class, ':', len(df_pos_sample))
    print('Majority class=', negative_class, ':', len(df_negatives))
    print('Proportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': 1')

    #SMOTE
    RANDOM_STATE = 42

    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = original.pop(class_var).values
    X = original.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(original.columns) + [class_var]
    df_smote.to_csv(f'../data/balancing/air_quality_{scalling}_train_SMOTEsampling.csv', index=False)

    smote_target_count = Series(smote_y).value_counts()
    print("SMOTE:")
    values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
    print('Minority class=', positive_class, ':', smote_target_count[positive_class])
    print('Majority class=', negative_class, ':', smote_target_count[negative_class])
    print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')
