

import os
import json
import argparse
import numpy as np
import pandas as pd
from pymatgen import Structure
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from pyemd import emd

from data_explore import composition_one_hot
from visualization import calc_obs_vs_pred, binarize_output

def read_and_trim_rdf(data, x_dir, trim='minimum'):
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


def krr_grid_search(alpha, gamma, X_data, y_data, test_size=0.2):
    kr = GridSearchCV( KernelRidge(),
                    scoring='neg_mean_absolute_error',
                    param_grid=[#{'kernel': ['rbf'], 'alpha': alpha, 'gamma': gamma},
                                {'kernel': ['linear'], 'alpha': alpha}]
                     )
    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=test_size, random_state=1) 
    kr = kr.fit(X_train, y_train)
    return kr.best_score_ , kr.best_params_, kr.cv_results_


def svr_grid_search(gamma, C, X_data, y_data, test_size=0.2):
    svr = GridSearchCV( SVR(),
                    scoring='neg_mean_absolute_error',
                    param_grid=[{'kernel': ['rbf'], 'gamma': gamma, 'C': C},
                                {'kernel': ['linear'], 'C': C}]
                      )
    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=test_size, random_state=1) 
    svr = svr.fit(X_train, y_train)
    return svr.best_score_ , svr.best_params_ #, svr.cv_results_


def calc_learning_curve(funct, X_data, y_data, test_size=0.2):
    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=test_size, random_state=1)
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


def int_or_str(value):
    '''
    For augment parser trim, where input can be either integer or string
    '''
    try:
        return int(value)
    except:
        return value


def earth_mover_dist(y_test, y_pred):
    '''
    '''
    dist = []
    dist_matrix = pd.read_csv('../similarity_matrix.csv', index_col='ionA').values
    dist_matrix = dist_matrix.copy(order='C')
    for i,y in enumerate(y_test):
        dist.append(emd(y.astype('float64'), y_pred[i].astype('float64'), 
                    dist_matrix.astype('float64')))

    return np.stack(dist).mean()


if __name__ == '__main__':
    # input parameters
    parser = argparse.ArgumentParser(description='Train machine learning algorithm',
                                    formatter_class=argparse.RawTextHelpFormatter)

    # input files of the training dataset
    parser.add_argument('--xdir', type=str, default='./',
                        help='All the rdf files for training')
    parser.add_argument('--input_file', type=str, default='../MP_modulus.json',
                        help='files contain target data')
    parser.add_argument('--funct', type=str, default='krr',
                        help='which function is used, currently support: \n' +
                            '   krr(Kernel Ridge Regression) \n' +
                            '   svm(Support Vector Machine) \n' +
                            '   rf(random forest) \n' +
                            '   lasso'
                        )
    parser.add_argument('--target', type=str, default='composition',
                        help='which properteis as target, currently support: \n' +
                            '   bulk_modulus \n' +
                            '   shear_modulus \n' +
                            '   density \n' +
                            '   formula: percentage of elements in each compound \n' +
                            '   composition: which types of elements in each compound \n' +
                            '   volume_per_atom \n' +
                            '   number_of_atoms: in the unit cell \n' +
                            '   type_of_elements: number of atomic species of the compound '
                        )
    parser.add_argument('--output', type=str, default='test_size_depend',
                        help='one of the following: \n'+
                            '   test_size_depend: change the test size from 0.1 to 0.9 \n' +
                            '   obs_vs_pred: plot the observation vs prediction results for \n' +
                            '       training and test set respectively, observation go first \n' +
                            '   confusion_maxtrix: histogram for multt-label output \n' +
                            '   grid_search: cross validation grid search for meta parameters \n' +
                            '   randon_guess: difference between ground truth and random guess/constant \n' +
                            '   learning_curve: the calculated learning curve will be stored \n' +
                            '       in ../learning_curve'
                        )
    parser.add_argument('--metrics', type=str, default='default',
                        help='which metrics is used, currently support: \n' +
                            '   default: mae for continuous data,  \n' +
                            '   mae: mean absolute error \n' +
                            '   mape: mean absolute pencentage error \n' +
                            '   emd: earth mover distance, i.e. wasserstein metrics'
                        )

    # parameters for rdf preprocessing
    parser.add_argument('--trim', type=str, default='minimum',
                        help='the number of shells for RDF')

    args = parser.parse_args()
    x_dir = args.xdir
    y_file = args.input_file
    target = args.target
    metrics_method = args.metrics
    funct_name = args.funct
    output = args.output
    trim = int_or_str(args.trim) 

    # prepare the dataset and split to train and test
    with open (y_file,'r') as f:
        data = json.load(f)
    if output != 'random_guess':
        X_data = read_and_trim_rdf(data, x_dir, trim=trim)

    # target_type can be continuous categorical ordinal
    # or multi-cont, multi-cate, multi-ord
    target_type = None

    # calculate the target values
    if target == 'bulk_modulus':
        target_type = 'continuous'
        y_data = np.array([ x['elasticity.K_VRH'] for x in data ])
        y_data = np.log10(y_data)
    elif target == 'shear_modulus':
        target_type = 'continuous'
        y_data = np.array([ x['elasticity.G_VRH'] for x in data ])
        y_data = np.log10(y_data)
    elif target == 'density':
        target_type = 'continuous'
        y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').density 
                            for x in data ])
        #np.savetxt('../density', y_data, delimiter=' ', fmt='%.3f')
    elif target == 'volume_per_atom':
        target_type = 'continuous'
        y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').volume 
                            / len(Structure.from_str(x['cif'], fmt='cif')) 
                            for x in data ])
    elif target == 'type_of_elements':
        target_type = 'ordinal'
        y_data = np.array([ len(Structure.from_str(x['cif'], fmt='cif').symbol_set) 
                            for x in data ])
        #target_type = 'categorical'
        #y_data = [ str(len(Structure.from_str(x['cif'], fmt='cif').symbol_set)) 
        #                    for x in data ]
    elif target == 'number_of_atoms':
        target_type = 'ordinal'
        y_data = np.array([ len(Structure.from_str(x['cif'], fmt='cif')) 
                            for x in data ])
    elif target == 'formula':
        target_type = 'multi-cont'
        y_data, elem_symbols = composition_one_hot(data=data, only_type=False)
        print(elem_symbols)
    elif target == 'composition':
        target_type = 'multi-cate'
        y_data, elem_symbols = composition_one_hot(data=data, only_type=True)
        print(elem_symbols)
    else:
        print('This target is not support, please check help')
        exit()

    # select the machine learning algorithm
    if funct_name == 'krr':
        funct = KernelRidge(alpha=1.0)
    elif funct_name == 'svm':
        if target_type in ('categorical', 'multi-cate'):
            funct = SVC(kernel='linear')
        else:
            funct = SVR(kernel='linear')
    elif funct_name == 'lasso':
        funct = Lasso()
    elif funct_name == 'rf':
        if target_type in ('categorical', 'multi-cate'):
            funct = RandomForestClassifier(max_depth=2, random_state=0)
        else:
            funct = RandomForestRegressor(max_depth=2, random_state=0)
    else:
        print('this algorithm is not support, please check help')
        exit()

    # use a 80/20 split except otherwise stated, e.g. in varying test_size
    test_size = 0.2
    if output == 'test_size_depend':
        for test_size in np.linspace(0.9, 0.1, 9):
            X_train, X_test, y_train, y_test = \
                train_test_split(X_data, y_data, test_size=test_size, random_state=1) 

            funct.fit(X_train, y_train)
            y_pred = funct.predict(X_test)

            if target_type == 'continuous':
                if metrics_method == 'default' or metrics == 'mae':
                    pred_acc = metrics.mean_absolute_error(y_test, y_pred)
                elif metrics_method == 'mape':
                    pred_acc = metrics.mean_absolute_percentage_error(y_test, y_pred)
                else:
                    print('This metrics is not support for continuous data')
            elif target_type == 'multi-cont':
                if metrics_method == 'default':
                    pred_acc = [ metrics.mean_absolute_error(y_test[:,i], y_pred[:,i])
                                for i in range(len(y_test[0])) ]
                elif metrics_method == 'emd':
                    pass
                else:
                    print('This metrics is not support for multi-continuous data')
            elif target_type == 'categorical'and target == 'type_of_elements':
                # now only implemented for type of elements
                y_pred[np.where(y_pred > 5)] = 5
                y_pred = list(map(str, list(np.int64(y_pred + 0.5))))
                #print(type(y_test), type(y_test[0]), y_test[0], type(y_pred), type(y_pred[0]), y_pred[0])
                pred_acc = metrics.accuracy_score(y_test, y_pred)
            elif target_type == 'multi-cate':
                if metrics_method == 'default':
                    pred_acc = [round(metrics.coverage_error(y_test, y_pred), 3),
                            round(metrics.label_ranking_average_precision_score(y_test, y_pred), 3),
                            round(metrics.label_ranking_loss(y_test, y_pred), 3) ]
                elif metrics_method == 'emd':
                    pred_acc = earth_mover_dist(y_test, y_pred)
                else:
                    print('This metrics is not support for multi-label data')
            elif target_type == 'ordinal':
                y_pred = np.int64(y_pred + 0.5)
                if metrics_method == 'default' or metrics == 'mae':
                    pred_acc = metrics.mean_absolute_error(y_test, y_pred)
                elif metrics_method == 'mape':
                    pred_acc = metrics.mean_absolute_percentage_error(y_test, y_pred)
                else:
                    print('This metrics is not support for ordinal data')
            else:
                print('target type not supported')

            print('Training size {} ; Training samples {} ; Metrics {}'.format(
                    round(1-test_size, 3), int((1-test_size)*len(y_data)), pred_acc))
    
    elif output == 'obs_vs_pred':
        calc_obs_vs_pred(funct=funct, X_data=X_data, y_data=y_data, test_size=test_size)
    
    elif output == 'confusion_matrix':
        if target_type == 'multi-cate':
            X_train, X_test, y_train, y_test = \
                    train_test_split(X_data, y_data, test_size=test_size, random_state=1)
            funct.fit(X_train, y_train)
            y_pred = funct.predict(X_test)
            # if the prediction is decmical, use this to round up
            y_bin = binarize_output(y_test, y_pred, threshold=0.4, save_to_file=False)

            # Calcuate label-wise (typically element-wise) confustion matrix for each label. 
            # The output is reshaped into a 'number of types of lables' by 4 
            # in the order of true-positive false-positive false-negative true-negative
            cm = metrics.multilabel_confusion_matrix(y_test, y_bin)
            np.savetxt('../confusion_matrix_' + str(round(test_size,3)), 
                        cm.reshape(len(y_pred[0]), 4), delimiter=' ', fmt='%.3f')
        elif target_type == 'categorical' and target == 'type_of_elements':
            # now only implemented for type of elements
            classes = list(map(str, list(range(1,6))))
            X_train, X_test, y_train, y_test = \
                    train_test_split(X_data, y_data, test_size=test_size, random_state=1)
            funct.fit(X_train, y_train)
            y_pred = funct.predict(X_test)
            y_pred[np.where(y_pred > 5)] = 5
            y_pred = list(map(str, list(np.int64(y_pred + 0.5))))
            cm = metrics.confusion_matrix(y_test, y_pred, labels=classes)
            np.savetxt('../confusion_matrix_multiclass', cm, delimiter=' ', fmt='%.3f')
        else:
            print('This target does not support confusion matrix')
    
    elif output == 'grid_search':
        if funct_name == 'krr':
            best_score, best_params, cv_results = \
                    krr_grid_search(alpha=np.logspace(-2, 2, 5), gamma=np.logspace(-2, 2, 5),
                                    X_data=X_data, y_data=y_data, test_size=test_size)
            print('best score {} ; best parameter {}'.format(best_score, best_params))
        elif funct_name == 'svm' and target_type not in ('categorical', 'multi-cate'):
            best_score, best_params, cv_results = \
                    svr_grid_search(gamma=np.logspace(-8, 1, 10), C=[1, 10, 100, 1000],
                                    X_data=X_data, y_data=y_data, test_size=test_size)
            print('best score {} ; best parameter {}'.format(best_score, best_params))
    
    elif output == 'learning_curve':
        calc_learning_curve(funct=funct, X_data=X_data, y_data=y_data, test_size=test_size)

    elif output == 'random_guess':
        nclass = 5
        max_class = 3

        y_pred_const = np.full(len(y_data), max_class)
        pred_acc_const = metrics.mean_absolute_error(y_data, y_pred_const)
        y_pred_rand = np.int64(np.random.rand(len(y_data)) * nclass + 1)
        pred_acc_rand = metrics.mean_absolute_error(y_data, y_pred_rand)

        print('constant {} ; random {}'.format(pred_acc_const, pred_acc_rand))

    else:
        print('This output is not supported')


