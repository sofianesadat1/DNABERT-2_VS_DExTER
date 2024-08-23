import os
import re

import numpy as np
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

import matplotlib.style
matplotlib.style.use('ggplot')

import operator

from numpy import matrix
import pandas as pd
from sklearn import linear_model

from scipy.stats.stats import pearsonr

NB_DISPLAYED_SELECTED_VARIABLES = 7


def run(training_file, testing_file):
    names = []
    training_x = []
    training_y = []
    with open(training_file, 'r') as infile:
        for line in infile.readlines():
            line = line.strip()
            if line.startswith('sequence'):
                names = re.split(r'\s+', line)[2:]
                continue
            else:
                tline = re.split(r'\s+', line)
                training_y.append(float(tline[1]))
                tmp = []
                for s in tline[2:]:
                    tmp.append(float(s))
                training_x.append(tmp)

    # lasso = linear_model.ElasticNetCV(l1_ratio=0.5, normalize=True, max_iter=1e6, fit_intercept=True, n_jobs=os.cpu_count(), cv=10)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    
    # Créer un pipeline qui inclut la normalisation des données
    lasso = make_pipeline(StandardScaler(), linear_model.LassoCV(max_iter=1000000, fit_intercept=True, n_jobs=-1, cv=10))

    lasso.fit(training_x, training_y)
    
    # Récupérer le modèle LassoCV du pipeline
    lasso_model = lasso.named_steps['lassocv']
    
    # Obtenir le nombre d'itérations effectuées par le modèle LassoCV
    nb_iter = lasso_model.n_iter_

    
    # Obtenir le coefficient de régularisation choisi par validation croisée
    alpha = lasso_model.alpha_

    coef = lasso_model.coef_

    testing_x = []
    testing_y = []
    testing_labels = []
    with open(testing_file, 'r') as infile:
        for line in infile.readlines():
            line = line.strip()
            if line.startswith('sequence'):
                continue
            else:
                tline = re.split(r'\s+', line)
                testing_labels.append(tline[0])
                testing_y.append(float(tline[1]))
                tmp = []
                for s in tline[2:]:
                    tmp.append(float(s))
                testing_x.append(tmp)

    predictions = lasso.predict(testing_x)

    mse = 0
    dct_errors = {}
    for index in range(len(testing_y)):
        error = testing_y[index] - predictions[index]
        dct_errors[testing_labels[index]] = float(error)
        mse += error**2
    mse /= len(testing_y)

    tmp = []
    for i in testing_y:
        tmp.append(float(i))
    pearson_correlation = pearsonr(tmp, predictions)  # (Pearson’s correlation coefficient,2-tailed p-value)
    pearson_correlation = pearson_correlation[0]

    score_r2 = lasso.score(testing_x, testing_y)

    # print('optimal alpha:', str(lasso.alpha_))

    df = pd.read_csv(training_file, delimiter='\s+', index_col=0, header=0)
    df = df._get_numeric_data()
    training_y = df[df.columns[0]].to_numpy()
    training_x = df[df.columns[1:]].to_numpy()
    alphas, _, coefs = linear_model.lars_path(training_x, training_y, method='lasso', verbose=True)
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    dct_rank_variable_selection = {}
    for index in range(len(names)):
        lst_alphas = list(coefs.tolist()[index])
        dct_rank_variable_selection[names[index]] = next((i for i, x in enumerate(lst_alphas) if x), None)
    dct_rank_variable_selection = {k: v for k, v in dct_rank_variable_selection.items() if v is not None}  # filter None values
    dct_rank_variable_selection = sorted(dct_rank_variable_selection.items(), key=operator.itemgetter(1))
    ordered_selected_variables = [name for name, rank in dct_rank_variable_selection]

    removables_variables = ordered_selected_variables[NB_DISPLAYED_SELECTED_VARIABLES:]
    removables_variables_indexes = []
    for rv in removables_variables:
        removables_variables_indexes.append(names.index(rv))

    coefs = np.delete(coefs, removables_variables_indexes, axis=0)

    f = plt.figure()
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(alpha*len(training_y), ymin, ymax, linestyle='solid', colors='r')
    plt.vlines(xx, ymin, ymax, linestyle='dashed', linewidth=0.5)
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.legend(ordered_selected_variables[:NB_DISPLAYED_SELECTED_VARIABLES], loc='best')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.show()
    b = zip(training_file, testing_file)
    c = [x[0] for x in b if x == (x[0],) * len(x)]
    result = "".join(c)
    result = result[:-6]
    f.savefig(result + '_lasso_path.pdf', bbox_inches='tight')

    pretty_model = pretty_print_linear(coef, names, dct_rank_variable_selection)

    return nb_iter, mse, pearson_correlation, score_r2, pretty_model, dct_errors


def pretty_print_linear(coefs, names, rvs):
    rvs = sorted(rvs, key=lambda x: x[0])
    ranks = [rank for _,rank in rvs]
    lst = zip(coefs, names, ranks)
    lst = sorted(lst, key=lambda x: np.abs(x[2]))
    ret = ''
    for coef, name, rank in lst:
        coef = round(coef, 2)
        if float(coef) != 0:
            ret += '(' + str(coef) + ' * %' + str(name) + ') + '
    ret = ret[:-3]
    return ret


def complete_run(training_file, testing_file):
    print('Computing LASSO...')
    nb_iter, mse1, correlation1, r21, pretty_model, _ = run(training_file, testing_file)
    print('\tnb_iter =', str(nb_iter))
    print('\tMSE =', str(mse1))
    print('\tCor =', str(correlation1))
    print('\tR2 =', str(r21))
    print('\t' + pretty_model)
    print('\n')


if __name__ == '__main__':
    complete_run(sys.argv[1], sys.argv[2])
