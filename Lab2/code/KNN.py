import numpy as np
from numpy import ndarray
from pandas import DataFrame, read_csv, unique, concat
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

file_tag = 'NYC_crashes'
filename = '../../data/firstDataset/NYC_collisions_tabular.csv'
target = 'PERSON_INJURY'
injured = 'Injured'
killed = 'Killed'
le = preprocessing.LabelEncoder()

# spliting into train and test
data: DataFrame = read_csv(filename)
values = {'Original': [len(data[data[target] == injured]), len(data[data[target] == killed])]}
#print(values)

y: np.ndarray = data.pop(target).values
x: np.ndarray = data.values
labels: np.ndarray = unique(y)
labels.sort()

trnX, tstX, trnY, tstY = train_test_split(x, y, train_size=0.7, stratify=y)
train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[target])], axis=1)
test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[target])], axis=1)

trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

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
        knn.fit(trnX, trnY)
        prdY = knn.predict(tstX)
        yvalues.append(accuracy_score(tstY, prdY))
        if yvalues[-1] > last_best:
            best = (n, d)
            last_best = yvalues[-1]
    values[d] = yvalues

figure()
multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
savefig('data/images/{file_tag}_knn_study.png')
show()
print('Best results with %d neighbors and %s' % (best[0], best[1]))
