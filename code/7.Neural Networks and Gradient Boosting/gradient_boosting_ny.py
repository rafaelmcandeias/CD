from numpy import ndarray, std, argsort
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import GradientBoostingClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT, plot_overfitting_study
from sklearn.metrics import accuracy_score

file_tag = 'NYC_collisions'
target = 'PERSON_INJURY'
dataset = 'firstDataset'

for scaling in ('minmax', 'zscore'):
    for balancing in ('undersampling', 'oversampling', 'SMOTEsampling'):
        train: DataFrame = read_csv(f'../../Lab4/balancing/{file_tag}_{scaling}_train_{balancing}.csv')
        trnY: ndarray = train.pop(target).values
        trnX: ndarray = train.values
        labels = unique(trnY)
        labels.sort()

        test: DataFrame = read_csv(f'../../data/{dataset}/{file_tag}_{scaling}_test.csv')
        tstY: ndarray = test.pop(target).values
        tstX: ndarray = test.values

        n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
        max_depths = [5, 10, 25]
        learning_rate = [.1, .5, .9]
        best = ('', 0, 0)
        last_best = 0
        best_model = None

        cols = len(max_depths)
        figure()
        fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
        for k in range(len(max_depths)):
            d = max_depths[k]
            values = {}
            for lr in learning_rate:
                yvalues = []
                for n in n_estimators:
                    gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                    gb.fit(trnX, trnY)
                    prdY = gb.predict(tstX)
                    yvalues.append(accuracy_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (d, lr, n)
                        last_best = yvalues[-1]
                        best_model = gb
                values[lr] = yvalues
            multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Gradient Boorsting with max_depth={d}',
                                xlabel='nr estimators', ylabel='accuracy', percentage=True)
        savefig(f'../data/images/{file_tag}_{scaling}_{balancing}_gb_study.png')
        print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

        # Best parameters
        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)
        plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
        savefig(f'../data/images/{file_tag}_{scaling}_{balancing}_gb_best.png')
        show()

        # features importance
        variables = train.columns
        importances = best_model.feature_importances_
        indices = argsort(importances)[::-1]
        stdevs = std([tree[0].feature_importances_ for tree in best_model.estimators_], axis=0)
        elems = []
        for f in range(len(variables)):
            elems += [variables[indices[f]]]
            print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

        figure()
        horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Gradient Boosting Features importance', xlabel='importance', ylabel='variables')
        savefig(f'../data/images/{file_tag}_{scaling}_{balancing}_gb_ranking.png')

        # plot overfitting study
        lr = 0.7
        max_depth = 10
        eval_metric = accuracy_score
        y_tst_values = []
        y_trn_values = []
        for n in n_estimators:
            gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
            gb.fit(trnX, trnY)
            prd_tst_Y = gb.predict(tstX)
            prd_trn_Y = gb.predict(trnX)
            y_tst_values.append(eval_metric(tstY, prd_tst_Y))
            y_trn_values.append(eval_metric(trnY, prd_trn_Y))
        plot_overfitting_study(n_estimators, y_trn_values, y_tst_values, name=f'GB_depth={max_depth}_lr={lr}', xlabel='nr_estimators', ylabel=str(eval_metric))
        savefig(f'../data/images/{file_tag}_{scaling}_{balancing}_gb_overfitting_study.png')