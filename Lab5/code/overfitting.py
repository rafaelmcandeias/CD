import numpy as np
from numpy import ndarray, argsort, arange
from pandas import DataFrame, read_csv, unique, concat
from matplotlib.pyplot import figure, subplots, savefig, show, Axes
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score


file_tag = 'NYC_collisions'
method = 'Smote'
#method = 'Overampling'
target = 'PERSON_INJURY'
"""""
data = read_csv('../../data/firstDataset/NYC_collisions_tabular_dummified.csv')
data = data.drop(['CRASH_DATE', 'CRASH_TIME'], axis=1)
X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
data_train = concat([X_train, y_train], axis=1)
data_train.to_csv('../../data/firstDataset/NYC_collisions_tabular_train.csv')
data_test = concat([X_test, y_test], axis=1)
data_test.to_csv('../../data/firstDataset/NYC_collisions_tabular_test.csv')
"""
train: DataFrame = read_csv('../../Lab4/data/balancing/NYC_collisions_train_SMOTEsampling.csv')
#train: DataFrame = read_csv('../../Lab4/data/balancing/NYC_collisions_train_oversampling.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv('../../Lab4/data/balancing/NYC_collisions_tabular_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

#fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)

max_depths = [2, 5, 10, 15, 20, 25]


imp = 0.0001
f = 'entropy'
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
#ytests = {}
#ytrains = {}
for d in max_depths:
    tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
    tree.fit(trnX, trnY)
    prdY = tree.predict(tstX)
    prd_tst_Y = tree.predict(tstX)
    prd_trn_Y = tree.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    #ytests[d] = y_tst_values
    #ytrains[d] = y_trn_values
figure()

plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth', ylabel=str(eval_metric))

savefig(f'../data/images/{file_tag}_{method}_overfitting_DT=imp{imp}_{f}.png')