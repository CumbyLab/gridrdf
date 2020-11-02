

import os
import json
import argparse
import numpy as np
from pymatgen import Structure
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from data_explore import composition_one_hot


def read_and_trim_rdf(data, x_dir, trim=100):
    '''
    Read the rdf files and trim them to the same length  
    for kernel methods training

    Args:
        data: modulus data from materials project
        x_dir: the dir has all (and only) the rdf files
        trim: must be one of 'no', 'minimum' or an integer
            no: no trim when the RDF already have the same length, 
            minimum: all rdf trim to the value of the smallest rdf
            integer value: if a value is given, all rdf longer than 
                this value will be trimmed, short than this value will  
                add 0.000 to the end

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

    # if the rdf is not 1d, make it 1d for machine learning input
    if len(all_rdf[0].shape) == 2:
        all_rdf = [ x.flatten() for x in all_rdf]
    
    return np.stack(all_rdf)


def krr_grid_search(X_train, y_train, alpha, gamma):
    kr = GridSearchCV( KernelRidge(),
                    scoring='neg_mean_absolute_error',
                    param_grid=[{'kernel': ['rbf'], 'alpha': alpha, 'gamma': gamma},
                                {'kernel': ['linear'], 'alpha': alpha}]
                     )
    kr = kr.fit(X_train, y_train)
    return kr.best_score_ , kr.best_params_, kr.cv_results_


def svr_grid_search(X_train, y_train, gamma, C):
    svr = GridSearchCV( SVR(),
                    scoring='neg_mean_absolute_error',
                    param_grid=[{'kernel': ['rbf'], 'gamma': gamma, 'C': C},
                                {'kernel': ['linear'], 'C': C}]
                      )
    svr = svr.fit(X_train, y_train)
    return svr.best_score_ , svr.best_params_ #, svr.cv_results_


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
    np.savetxt('../learning_curve', learning_curve_results, delimiter=' ', fmt='%.3f')


def calc_obs_vs_pred(funct, X_train, X_test, y_train, y_test):
    test_size = round( len(X_test) / (len(X_train) + len(X_test) ), 3)
    y_pred_train = funct.predict(X_train)
    np.savetxt('../test.' + str(test_size), 
                np.stack([y_test, y_pred]).transpose(), 
                delimiter=' ', fmt='%.3f')
    np.savetxt('../train.' + str(test_size), 
                np.stack([y_train, y_pred_train]).transpose(), 
                delimiter=' ', fmt='%.3f')


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
    parser.add_argument('--yfile', type=str, default='../MP_modulus.json',
                        help='bulk modulus values as target')
    parser.add_argument('--funct', type=str, default='krr',
                        help='''which function is used, currently support krr(Kernel Ridge Regression)
                        svm(Support Vector Machine), rf(random forest), lasso''')
    parser.add_argument('--target', type=str, default='composition',
                        help='''which properteis as target, currently support bulk_modulus,
                        shear_modulus, density, composition''')

    # parameters for rdf preprocessing
    parser.add_argument('--trim', type=str, default='minimum',
                        help='the number of shells for RDF')

    # parameters for the training process
    parser.add_argument('--test_size_depend', dest='test_size_depend', action='store_true',
                        help='change the test size from 0.1 to 0.9')
    parser.set_defaults(test_size_depend=False)
    parser.add_argument('--obs_vs_pred', dest='obs_vs_pred', action='store_true',
                        help='''plot the observation vs prediction results for training
                        and test set respectively, observation go first ''')
    parser.set_defaults(obs_vs_pred=False)
    parser.add_argument('--grid_search', dest='grid_search', action='store_true', 
                        help='perform cross validation grid search for meta parameters')
    parser.set_defaults(grid_search=False)
    parser.add_argument('--learning_curve', dest='learning_curve', action='store_true', 
                        help='the calculated learning curve will be stored in ../learning_curve')
    parser.set_defaults(learning_curve=False)
    
    args = parser.parse_args()
    x_dir = args.xdir
    y_file = args.yfile
    target = args.target
    funct_name = args.funct
    trim = int_or_str(args.trim) 

    grid_search = args.grid_search
    learning_curve = args.learning_curve
    obs_vs_pred = args.obs_vs_pred
    test_size_depend = args.test_size_depend

    # prepare the dataset and split to train and test
    with open (y_file,'r') as f:
        data = json.load(f)
    X_data = read_and_trim_rdf(data, x_dir, trim=trim)

    # specify the target 
    if target == 'bulk_modulus':
        y_data = np.array([ x['elasticity.K_VRH'] for x in data ])
        y_data = np.log10(y_data)
    elif target == 'shear_modulus':
        y_data = np.array([ x['elasticity.G_VRH'] for x in data ])
        y_data = np.log10(y_data)
    elif target == 'density':
        y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').density 
                            for x in data ])
        #np.savetxt('../density', y_data, delimiter=' ', fmt='%.3f')
    elif target == 'vol_per_atom':
        y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').volume 
                            / len(Structure.from_str(x['cif'], fmt='cif')) 
                            for x in data ])
    elif target == 'composition':
        only_type = True
        if only_type:
            classifier = True
        y_data, elem_symbols = composition_one_hot(data=data, only_type=only_type)
        print(elem_symbols)
    else:
        print('this target is not support, please check help')
        exit()

    # select the machine learning algorithm
    if funct_name == 'krr':
        funct = KernelRidge(alpha=1.0)
    elif funct_name == 'svm':
        if classifier:
            funct = SVC(kernel='linear')
        else:
            funct = SVR(kernel='linear')
    elif funct_name == 'lasso':
        funct = Lasso()
    elif funct_name == 'rf':
        if classifier:
            funct = RandomForestClassifier(max_depth=2, random_state=0)
        else:
            funct = RandomForestRegressor(max_depth=2, random_state=0)
    else:
        print('this algorithm is not support, please check help')
        exit()

    if test_size_depend:
        for test_size in np.linspace(0.9, 0.1, 9):
            X_train, X_test, y_train, y_test = \
                train_test_split(X_data, y_data, test_size=test_size, random_state=1) 

            funct.fit(X_train, y_train)
            y_pred = funct.predict(X_test)
            if target == 'composition':
                if only_type:
                    pred_acc = [round(metrics.coverage_error(y_test, y_pred), 3),
                                round(metrics.label_ranking_average_precision_score(y_test, y_pred), 3),
                                round(metrics.label_ranking_loss(y_test, y_pred), 3) ]
                else:
                    pred_acc = [ metrics.mean_absolute_error(y_test[:,i], y_pred[:,i])
                                for i in range(len(y_test[0])) ]
            else:
                pred_acc = metrics.mean_absolute_error(y_test, y_pred)

            print('Training size {} ; Training samples {} ; Metrics {}'.format(
                    round(1-test_size, 3), int((1-test_size)*len(y_data)), pred_acc))

            # output the observation vs prediction plot
            if obs_vs_pred:
                calc_obs_vs_pred(funct=funct, X_train=X_train, X_test=X_test, 
                                y_train=y_train, y_test=y_test)

    if grid_search:
        test_size = 0.2
        X_train, X_test, y_train, y_test = \
            train_test_split(X_data, y_data, test_size=test_size, random_state=1) 
        #krr_grid_search(X_train=X_train, y_train=y_train,
        #                alpha=[1e0, 0.1, 1e-2, 1e-3], gamma=np.logspace(-2, 2, 5))
        svr_grid_search(X_train=X_train, y_train=y_train, 
                        gamma=np.logspace(-8, 1, 10), C=[1, 10, 100, 1000])

    if learning_curve:
        test_size = 0.2
        X_train, X_test, y_train, y_test = \
            train_test_split(X_data, y_data, test_size=test_size, random_state=1) 
        calc_learning_curve(funct=funct, X_train=X_train, y_train=y_train)

