from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig
from ds_charts import plot_evaluation_results, bar_chart

target = 'PERSON_INJURY'

for scalling in ("minmax", "zscore"):
    for balancing in ('undersampling', 'SMOTEsampling', 'oversampling'):
        train: DataFrame = read_csv(f'../data/balancing/NYC_collisions_{scalling}_train_{balancing}.csv')
        trnY: ndarray = train.pop(target).values
        trnX: ndarray = train.values
        labels = unique(trnY)
        labels.sort()

        test: DataFrame = read_csv(f'../../data/firstDataset/NYC_collisions_{scalling}_test.csv')
        tstY: ndarray = test.pop(target).values
        tstX: ndarray = test.values

        trnX[trnX < 0] = 0
        tstX[tstX < 0] = 0

        clf = GaussianNB()
        clf.fit(trnX, trnY)
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
        savefig(f'../data/nb/NYC_collisions_{scalling}_{balancing}_best.png')

        estimators = {'GaussianNB': GaussianNB(),
                    'MultinomialNB': MultinomialNB(),
                    'BernoulliNB': BernoulliNB()
                    }
        xvalues = []
        yvalues = []
        for clf in estimators:
            xvalues.append(clf)
            estimators[clf].fit(trnX, trnY)
            prdY = estimators[clf].predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))

        figure()
        bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
        savefig(f'../data/nb/NYC_collisions_{scalling}_{balancing}_study.png')
