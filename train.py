

import os
import json
import argparse
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import learning_curve, GridSearchCV


def read_and_trim_rdf(x_dir):
    '''
    Read the rdf files and trim them to the same length  
    for kernel methods training

    Args:
        x_dir: the dir has all (and only) the rdf files
    Return:
        a two dimensional matrix, first dimension is the number of
        samples, and the second dimension is the flatten 1D rdf
    '''
    all_rdf = []
    rdf_len = []
    for rdf_file in os.listdir(x_dir):
        rdf = np.loadtxt(rdf_file, delimiter=',')
        all_rdf.append(rdf)
        rdf_len.append(len(rdf))

    min_len = np.array(rdf_len).min()
    all_rdf = [ x[:min_len].flatten() for x in all_rdf]
    return np.stack(all_rdf)


if __name__ == '__main__':
    # input parameters
    parse = argparse.ArgumentParser(description='Train machine learning algorithm')
    parse.add_argument('--xdir', type=str, default='./',
                        help='All the rdf files for training')
    parse.add_argument('--yfile', type=str, default='../bm.json',
                        help='bulk modulus values as target')
    parse.add_argument('--output', type=str, default='../learning_curve',
                        help='output files contain the learning curve')
    args = parse.parse_args()
    x_dir = args.xdir
    y_file = args.yfile
    outfile = args.output

    # prepare the dataset and split to train and test
    X_data = read_and_trim_rdf(x_dir)
    print(X_data.shape)
    with open (y_file,'r') as f:
        d = json.load(f)
    y_data = np.array([ x['elasticity.K_Voigt'] for x in d ])
    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=0.2, random_state=1) 

    # grid search meta parameters
    grid_search_para = True
    if grid_search_para:
        kr = GridSearchCV( KernelRidge(kernel='linear'),
                            scoring='neg_mean_absolute_error',
                            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]} )
        kr = kr.fit(X_train, y_train)
        print(kr.best_score_ , kr.best_params_)
        print(kr.cv_results_)


    # get the learning curve
    calc_learning_curve = False
    if calc_learning_curve:
        pipe_krr = Pipeline([ #('scl', StandardScaler()),
                            ('krr', KernelRidge(alpha=1.0)), ])

        train_sizes, train_scores, test_scores = \
            learning_curve(estimator=pipe_krr, X=X_train, y=y_train, 
                            train_sizes=np.linspace(0.1, 1.0, 10),
                            scoring='neg_mean_absolute_error',
                            cv=10, n_jobs=1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        learning_curve_results = np.stack([ train_mean, train_std, test_mean, test_std ])
        learning_curve_results = learning_curve_results.transpose()
        np.savetxt(outfile, learning_curve_results, delimiter=' ', fmt='%.3f')


    if False:
        # this is the original fitting, no longer use
        clf = KernelRidge(alpha=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(metrics.mean_absolute_error(y_test, y_pred))



