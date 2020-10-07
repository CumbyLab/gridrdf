

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


def read_and_trim_rdf(x_dir, trim=True, make_1d=True):
    '''
    Read the rdf files and trim them to the same length  
    for kernel methods training

    Args:
        x_dir: the dir has all (and only) the rdf files
        trim: set to False if the RDF already have the same length, 
            or not necessary to make them the same length
        make_1d: if the rdf is not 1d, make it 1d for machine
            learning input
    Return:
        a two dimensional matrix, first dimension is the number of
        samples, and the second dimension is the flatten 1D rdf
    '''
    all_rdf = []
    rdf_len = []
    for rdf_file in os.listdir(x_dir):
        rdf = np.loadtxt(rdf_file, delimiter=' ')
        all_rdf.append(rdf)
        rdf_len.append(len(rdf))

    if trim:
        min_len = np.array(rdf_len).min()
        all_rdf = [ x[:min_len] for x in all_rdf]
    
    if make_1d:
        all_rdf = [ x.flatten() for x in all_rdf]
    
    return np.stack(all_rdf)


if __name__ == '__main__':
    # input parameters
    parser = argparse.ArgumentParser(description='Train machine learning algorithm')

    # input files of the training dataset
    parser.add_argument('--xdir', type=str, default='./',
                        help='All the rdf files for training')
    parser.add_argument('--yfile', type=str, default='../bm.json',
                        help='bulk modulus values as target')
    parser.add_argument('--output', type=str, default='../learning_curve',
                        help='output files contain the learning curve')

    # parameters for rdf preprocessing
    parser.add_argument('--trim', dest='trim', action='store_true')
    parser.add_argument('--no-trim', dest='trim', action='store_false')
    parser.set_defaults(trim=True)

    parser.add_argument('--make_1d', dest='make_1d', action='store_true')
    parser.add_argument('--no-make_1d', dest='make_1d', action='store_false')
    parser.set_defaults(make_1d=True)

    # parameters for the training process
    parser.add_argument('--grid_search', dest='grid_search', action='store_true')
    parser.add_argument('--no-grid_search', dest='grid_search', action='store_false')
    parser.set_defaults(grid_search=False)

    parser.add_argument('--learning_curve', dest='learning_curve', action='store_true')
    parser.add_argument('--no-learning_curve', dest='learning_curve', action='store_false')
    parser.set_defaults(learning_curve=False)

    args = parser.parse_args()
    x_dir = args.xdir
    y_file = args.yfile

    trim = args.trim
    make_1d = args.make_1d

    grid_search_para = args.grid_search
    calc_learning_curve = args.learning_curve
    outfile = args.output

    # prepare the dataset and split to train and test
    X_data = read_and_trim_rdf(x_dir, trim=trim, make_1d=make_1d)

    with open (y_file,'r') as f:
        data = json.load(f)
    y_data = np.array([ x['elasticity.K_Voigt'] for x in data ])
    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=0.2, random_state=1) 

    # grid search meta parameters
    if grid_search_para:
        kr = GridSearchCV( KernelRidge(),
                            scoring='neg_mean_absolute_error',
                            param_grid=[ {'kernel': ['rbf'],
                                            'alpha': [1e0, 0.1, 1e-2, 1e-3],
                                            'gamma': np.logspace(-2, 2, 5)},
                                            {'kernel': ['linear'],
                                            'alpha': [1e0, 0.1, 1e-2, 1e-3]},
                                        ] )

        kr = kr.fit(X_train, y_train)
        print(kr.best_score_ , kr.best_params_)
        print(kr.cv_results_)

    # get the learning curve
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



