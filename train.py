

import os
import json
import argparse
import numpy as np
from pymatgen import Structure
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def read_and_trim_rdf(data, x_dir, trim=100, make_1d=True):
    '''
    Read the rdf files and trim them to the same length  
    for kernel methods training

    Args:
        data: modulus data from materials project
        x_dir: the dir has all (and only) the rdf files
        trim: must be one of 'None', 'min' or an integer
            no: no trim when the RDF already have the same length, 
            minimum: all rdf trim to the value of the smallest rdf
            integer value: if a value is given, all rdf longer than 
                this value will be trimmed, short than this value will  
                add 0.000 to the end
        make_1d: if the rdf is not 1d, make it 1d for machine
            learning input
    Return:
        a two dimensional matrix, first dimension is the number of
        samples, and the second dimension is the flatten 1D rdf
    '''
    all_rdf = []
    rdf_lens = []
    for d in data:
        rdf_file = d['task_id']
        rdf = np.loadtxt(rdf_file, delimiter=' ')
        all_rdf.append(rdf)
        rdf_lens.append(len(rdf))

    if trim == 'minimum':
        min_len = np.array(rdf_lens).min()
        all_rdf = [ x[:min_len] for x in all_rdf]
    elif isinstance(trim, int):
        for i, rdf_len in enumerate(rdf_lens):
            if rdf_len < trim:
                nbins = len(all_rdf[i][0])
                all_rdf[i] = np.append( all_rdf[i], 
                                        [[0.0] * nbins] * (trim-rdf_len), 
                                        axis=0 )
            else:
                all_rdf[i] = all_rdf[i][:trim]
    elif trim == 'no':
        pass 
    else:
        print('wrong value provided for trim') 

    if make_1d:
        all_rdf = [ x.flatten() for x in all_rdf]
    
    return np.stack(all_rdf)


def krr_grid_search(X_train, y_train):
    kr = GridSearchCV( sklearn.kernel_ridge.KernelRidge(),
                    scoring='neg_mean_absolute_error',
                    param_grid=[ {'kernel': ['rbf'],
                                    'alpha': [1e0, 0.1, 1e-2, 1e-3],
                                    'gamma': np.logspace(-2, 2, 5)},
                                    {'kernel': ['linear'],
                                    'alpha': [1e0, 0.1, 1e-2, 1e-3]},
                                ] )
    kr = kr.fit(X_train, y_train)
    return kr.best_score_ , kr.best_params_, kr.cv_results_


def calc_learning_curve(funct, X_train, y_train):
    pipe = Pipeline([ #('scl', StandardScaler()),
                    ('krr', funct), ])

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator=pipe, X=X_train, y=y_train, 
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


def int_or_str(value):
    try:
        return int(value)
    except:
        return value


if __name__ == '__main__':
    # input parameters
    parser = argparse.ArgumentParser(description='Train machine learning algorithm')

    # input files of the training dataset
    parser.add_argument('--xdir', type=str, default='./',
                        help='All the rdf files for training')
    parser.add_argument('--yfile', type=str, default='../mp_icsd.json',
                        help='bulk modulus values as target')
    parser.add_argument('--output', type=str, default='../learning_curve',
                        help='output files contain the learning curve')

    # parameters for rdf preprocessing
    parser.add_argument('--trim', type=str, default='100',
                        help='the number of shells for RDF')
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
    trim = int_or_str(args.trim) 
    make_1d = args.make_1d

    grid_search_para = args.grid_search
    learning_curve = args.learning_curve
    outfile = args.output

    # prepare the dataset and split to train and test

    with open (y_file,'r') as f:
        data = json.load(f)
    X_data = read_and_trim_rdf(data, x_dir, trim=trim, make_1d=make_1d)

    #y_data = np.array([ x['elasticity.K_VRH'] for x in data ])
    #y_data = np.log10(y_data)
    y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').density for x in data ])
    np.savetxt('../density', y_data, delimiter=' ', fmt='%.3f')

    for test_size in np.linspace(0.1, 0.9, 9):
        X_train, X_test, y_train, y_test = \
            train_test_split(X_data, y_data, test_size=test_size, random_state=1) 

        clf = KernelRidge(alpha=1.0)
        #clf = SVR(kernel='linear')
        #clf = Lasso()
        #clf = RandomForestRegressor(max_depth=2, random_state=0)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(metrics.mean_absolute_error(y_test, y_pred))

        # output the observation vs predictin plot
        obs_vs_pred = False
        if obs_vs_pred: 
            y_pred_train = clf.predict(X_train)
            np.savetxt('../test.'+str(test_size), 
                        np.stack([y_test,y_pred]).transpose(), 
                        delimiter=' ', fmt='%.3f')
            np.savetxt('../train.'+str(test_size), 
                        np.stack([y_train,y_pred_train]).transpose(), 
                        delimiter=' ', fmt='%.3f')

    if grid_search_para:
        krr_grid_search(X_train=X_train, y_train=y_train)

    if learning_curve:
        calc_learning_curve(funct=sklearn.kernel_ridge.KernelRidge(alpha=1.0),
                            X_train=X_train, y_train=y_train)







