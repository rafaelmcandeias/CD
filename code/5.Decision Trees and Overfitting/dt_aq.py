from numpy import ndarray, argsort
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig
from sklearn.tree import DecisionTreeClassifier, plot_tree
from ds_charts import plot_evaluation_results, multiple_line_chart
from sklearn.metrics import accuracy_score
from ds_charts import horizontal_bar_chart

file_tag = 'air_quality'
target = 'ALARM'

for scalling in ('minmax', 'zscore'):
    for balancing in ('undersampling', 'oversampling', 'SMOTEsampling'):

        train: DataFrame = read_csv(f'../../Lab4/data/balancing/{file_tag}_{scalling}_train_{balancing}.csv')
        trnY: ndarray = train.pop(target).values
        trnX: ndarray = train.values
        labels = unique(trnY)
        labels.sort()

        test: DataFrame = read_csv(f'../../data/secondDataset/{file_tag}_{scalling}_test.csv')
        tstY: ndarray = test.pop(target).values
        tstX: ndarray = test.values


        min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
        max_depths = [2, 5, 10, 15, 20, 25]
        criteria = ['entropy', 'gini']
        best = ('',  0, 0.0)
        last_best = 0
        best_model = None

        figure()
        fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)

        for k in range(len(criteria)):
            f = criteria[k]
            values = {}
            for d in max_depths:
                yvalues = []
                for imp in min_impurity_decrease:
                    tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                    tree.fit(trnX, trnY)
                    prdY = tree.predict(tstX)
                    yvalues.append(accuracy_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (f, d, imp)
                        print(best)
                        last_best = yvalues[-1]
                        best_model = tree

                values[d] = yvalues
            multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                                xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
        savefig(f'../data/images/{file_tag}_{scalling}_{balancing}_dt_study.png')
        print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

        labels = [str(value) for value in labels]
        plot_tree(best_model, feature_names=train.columns, class_names=labels)
        savefig(f'../data/images/{file_tag}_{scalling}_{balancing}_dt_best_tree.png')

        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
        savefig(f'../data/images/{file_tag}_{scalling}_{balancing}_dt_best.png')

        variables = train.columns
        importances = best_model.feature_importances_
        indices = argsort(importances)[::-1]
        elems = []
        imp_values = []
        for f in range(len(variables)):
            elems += [variables[indices[f]]]
            imp_values += [importances[indices[f]]]
            print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

        figure()
        horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
        savefig(f'../data/images/{file_tag}_{scalling}_{balancing}_dt_ranking.png')
