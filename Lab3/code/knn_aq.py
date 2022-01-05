from numpy import ndarray
from pandas import DataFrame, read_csv, unique, concat
from matplotlib.pyplot import figure, savefig
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

file_tag = 'air_quality'
target = 'ALARM'

data = read_csv('../../data/secondDataset/air_quality_tabular_dummified.csv')
X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
data_train = concat([X_train, y_train], axis=1)
data_train.to_csv('../../data/secondDataset/air_quality_tabular_train.csv')
data_test = concat([X_test, y_test], axis=1)
data_test.to_csv('../../data/secondDataset/air_quality_tabular_test.csv')

train: DataFrame = read_csv('../../data/secondDataset/air_quality_tabular_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv('../../data/secondDataset/air_quality_tabular_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}
best = (0, '')
last_best = 0
for d in dist:
    yvalues = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(X_train, y_train)
        prdY = knn.predict(X_test)
        yvalues.append(accuracy_score(y_test, prdY))
        if yvalues[-1] > last_best:
            best = (n, d)
            last_best = yvalues[-1]
    values[d] = yvalues

figure()
multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
savefig('../data/knn/images/air_quality_knn_study.png')
print('Best results with %d neighbors and %s' % (best[0], best[1]))

clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig('../data/knn/images/air_quality_knn_best.png')
