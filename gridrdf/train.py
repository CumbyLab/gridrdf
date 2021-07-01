""" Train machine learning models using GRID and RDF descriptors.

Utility functions to simplify training of general ML models using
GRID, composition and different dissimilarity measures (including
Earth mover's distance).
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pymatgen import Structure
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from pyemd import emd

from .data_explore import rdf_trim, rdf_flatten, batch_shell_similarity, batch_lattice
from .composition import composition_one_hot, bonding_matrix
from .visualization import calc_obs_vs_pred, binarize_output, n_best_middle_worst
from .extendRDF import shell_similarity
from .data_io import rdf_read, shell_similarity_read
from .misc import int_or_str


def train_test_split_2D(X_data, y_data, test_size, random_state=1):
    """ Modified version of train_test_split designed to split a square nxn matrix correctly. """
    
    print('Using 2D split')
    # Turn things (back?) into a DataFrame so we retain the indices
    full_X = pd.DataFrame(X_data)
    full_y = pd.DataFrame(y_data)
    
    X_train, X_test, y_train, y_test = \
        train_test_split_old(full_X, full_y, test_size=test_size, random_state=1) 
        
    X_train = X_train.reindex(X_train.index, axis=1).to_numpy()
    X_test = X_test.drop(X_test.index, axis=1).to_numpy()
    
    # If only one training column, we need to call to_numpy differently
    # to avoid including indices in the output
    if y_train.shape[1] > 1:
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
    else:
        y_train = y_train[0].to_numpy()
        y_test = y_test[0].to_numpy()

    return X_train, X_test, y_train, y_test
    
def calc_obs_vs_pred_2D(funct, X_data, y_data, test_size, outdir='./'):
    '''
    The observation vs prediction plot for 2D X data (e.g. similarity matrix)
    '''
    X_train, X_test, y_train, y_test = \
            train_test_split_2D(X_data, y_data, test_size=test_size, random_state=1)
    funct.fit(X_train, y_train)
    y_pred = funct.predict(X_test)
    y_pred_train = funct.predict(X_train)
    
    np.savetxt(os.path.normpath(os.path.join(outdir, 'test.' + str(test_size))), 
                np.stack([y_test, y_pred]).transpose(), 
                delimiter=' ', fmt='%.3f')
    np.savetxt(os.path.normpath(os.path.join(outdir, 'train.' + str(test_size))), 
                np.stack([y_train, y_pred_train]).transpose(), 
                delimiter=' ', fmt='%.3f')

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

    print('Splitting data ... ', end='')
    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=test_size, random_state=1)
        
    print('Done')
    print('Training learning curve ... ', end='')
    pipe = Pipeline([ #('scl', StandardScaler()),
                ('krr', funct), ])
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator=pipe, X=X_train, y=y_train, 
                        train_sizes=np.linspace(0.1, 1.0, 10),
                        scoring='neg_mean_absolute_error',
                        cv=10, n_jobs=1)

    print('Done')
    print(train_sizes, train_scores, test_scores)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    learning_curve_results = np.stack([ train_mean, train_std, test_mean, test_std ])
    learning_curve_results = learning_curve_results.transpose()
    print(learning_curve_results.shape)
    np.savetxt(os.path.normpath(os.path.join(output_dir, 'learning_curve')), learning_curve_results, delimiter=' ', fmt='%.3f')


def emd_of_two_compositions(y_test, y_pred, pettifor_index=True):
    '''
    Calcalate the earth mover's distance of two compositions, the composition should be 
    a 78-element vector, as the similarity_matrix is 78x78 matrix in the order of  
    atom numbers

    Args:
        y_test: test data with dimension n*78, the second dimension is by default in the  
            pettifor order, see 'composition_one_hot' function in data_explore.py
        y_pred: prediction data
        pettifor_index: whether transform the element order from the peroidic table
            number to Pettifor number
    Return:

    '''
    dist = []
    elem_similarity_file = os.path.join(sys.path[0], 'similarity_matrix.csv')
    dist_matrix = pd.read_csv(elem_similarity_file, index_col='ionA')
    dist_matrix = 1 / (np.log10(1 / dist_matrix + 1))

    if pettifor_index:
        pettifor = ['Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Yb', 'Eu', 'Y',  'Sc', 'Lu', 'Tm', 'Er', 'Ho', 
            'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 'Zr', 'Hf', 'Ti', 'Nb', 'Ta', 'V',  'Mo', 
            'W',  'Cr', 'Tc', 'Re', 'Mn', 'Fe', 'Os', 'Ru', 'Co', 'Ir', 'Rh', 'Ni', 'Pt', 'Pd', 'Au', 'Ag', 
            'Cu', 'Mg', 'Hg', 'Cd', 'Zn', 'Be', 'Tl', 'In', 'Al', 'Ga', 'Pb', 'Sn', 'Ge', 'Si', 'B',  'Bi', 
            'Sb', 'As', 'P',  'Te', 'Se', 'S', 'C', 'I', 'Br', 'Cl', 'N', 'O', 'F', 'H']
        dist_matrix = dist_matrix.reindex(columns=pettifor, index=pettifor) 
    dist_matrix = dist_matrix.values

    # the earth mover's distance package need order C and float64
    dist_matrix = dist_matrix.copy(order='C')
    for y_t, y_p in zip(y_test, y_pred):
        dist.append(emd(y_t.astype('float64'), y_p.astype('float64'), 
                    dist_matrix.astype('float64')))
    
    #print(len(y_pred), np.count_nonzero(np.array(dist)))    
    return np.stack(dist)


def bond_to_atom(data, nelem=78):
    '''
    Convert atomic pairs bonding into the exsitance of elements

    Args:
        Data: y_data of flatten matrix
        nelem: number of elements in the periodic table, determines the shape
            of the bonding matrix
    Return:
        One-hot vector of elements
    '''
    new_data = []
    for y in data:
        y = np.reshape(y, (nelem, nelem))
        # make all the non-zero values to 1
        # NOTETHAT the sum method is only valid for a symmetric bonding matrix
        new_data.append(np.sum(y, axis=0).astype(bool).astype(int))
    return np.stack(new_data)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train machine learning algorithm',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--rdf_dir', type=str, default='./',
                        help='The dir of all rdf files for training \n' +
                            '   default is current work dir'
                        )
    parser.add_argument('--input_file', type=str, 
                        default='../MP_modulus_datasets/MP_modulus.json',
                        help='files contain target data \n' +
                            '   default is ../MP_modulus.json'
                        )
    parser.add_argument('--input_features', type=str, default='extended_rdf',
                        help='features used for machine learning: \n' +
                            '   extended_rdf  \n' +
                            '   shell_similarity: shell-wise similarity values \n' +
                            '   fourier_space:  \n' +
                            '   lattice_abc: a, b, c, alpha, beta, gamma \n' +
                            '   lattice_matrix: a 3x3 matrix of lattice \n' +
                            '   formula:  \n' +
                            '   composition:  \n' +
                            '   distance_matrix: precomputed pair-wise distances (e.g. EMD) \n' +
                            ' '
                        )
    parser.add_argument('--output_dir', type=str, default = './',
                        help='directory to put output files'
                        )
    parser.add_argument('--funct', type=str, default='krr',
                        help='which function is used, currently support: \n' +
                            '   krr(default, Kernel Ridge Regression) \n' +
                            '   svm(Support Vector Machine) \n' +
                            '   rf(random forest) \n' +
                            '   lasso: linear model with L1 regulation \n' +
                            '   elastic_net: both L1 and L2 regulation are used \n' + 
                            '   knn_reg: KNeighborsRegressor'
                        )
    parser.add_argument('--target', type=str, default='bulk_modulus',
                        help='which properteis as target, currently support: \n' +
                            '   bulk_modulus \n' +
                            '   shear_modulus \n' +
                            '   density \n' +
                            '   composition: percentage of elements in each compound \n' +
                            '   type_of_elements: which types of elements in each compound \n' +
                            '   volume_per_atom \n' +
                            '   volume \n' +
                            '   space_group_number \n' +
                            '   average_coordination \n' +
                            '   number_of_atoms: in the unit cell \n' +
                            '   bonding_type:  \n' +
                            '   average_coordination:  \n' +
                            '   average_bond_length:  \n' +
                            '   bond_length_std:  \n' +
                            '   ave_bond_std:  \n' +
                            '   coord_num_std:  \n' +
                            '   num_sg_operation:  \n' +
                            '   number_of_species: number of atomic species of the compound \n' +
                            '   band_gap: band gap data contained in input_file (i.e. from materials project)'
                        )
    parser.add_argument('--log_target', type=bool, default=False, 
                        help='whether to take log of target before fitting (default False)'
                        )
    parser.add_argument('--task', type=str, default='test_size_depend',
                        help='one of the following: \n'+
                            '   test_size_depend: default, change the test size from 0.1 to 0.9 \n' +
                            '   obs_vs_pred: plot the observation vs prediction results for \n' +
                            '       training and test set respectively, observation go first \n' +
                            '   confusion_maxtrix: histogram for multt-label output \n' +
                            '   grid_search: cross validation grid search for meta parameters \n' +
                            '   randon_guess: difference between ground truth and random guess/constant \n' +
                            '   emd_visual:  \n' +
                            '   learning_curve: the calculated learning curve will be stored \n' +
                            '       in output_dir/learning_curve'
                        )
    parser.add_argument('--metrics', type=str, default='default',
                        help='which metrics is used, currently support: \n' +
                            '   default: mae for continuous data \n' +
                            '   mae: mean absolute error \n' +
                            '   mape: mean absolute pencentage error \n' +
                            '   emd: earth mover distance, i.e. wasserstein metrics'
                        )
    parser.add_argument('--trim', type=str, default='minimum',
                        help='the number of shells for RDF used as input feature\n' +
                            '   minimum: default, use the minimum number of shells \n' +
                            '   none: no trim, suitable for origin RDF  \n' +
                            '   or an integer number'
                        )
    parser.add_argument('--dist_matrix', type=str, default=None,
                        help = 'file containing pre-computed pairwise distances to use \n' +
                               'instead of Sklearn distance metric (e.g. EMD)'
                        )
                        
                        
    args = parser.parse_args()
    rdf_dir = args.rdf_dir
    input_file = args.input_file
    target = args.target
    metrics_method = args.metrics
    funct_name = args.funct
    task = args.task
    trim = int_or_str(args.trim)
    input_features = args.input_features.split()
    dist_matrix = args.dist_matrix
    output_dir = args.output_dir
    
    

    print('Reading data input ... ', end=''),

    # read the dataset from input file
    with open (input_file,'r') as f:
        data = json.load(f)
        
    print('Done')

    # the following input features are prepared
    # if the task is to give randon guess (for comparsion with prediction)
    # then no input features is needed
    # otherwise the input features are calculated and comibined
    if task != 'random_guess':
        all_features = [
            'extended_rdf', 'shell_similarity', 
            'fourier_space', 
            'lattice_abc', 'lattice_matrix',
            'formula', 'composition',
            'distance_matrix',
        ]
        if not set(input_features).issubset(all_features):
            print('Wrong feature append argument')

        X_data = None
        if 'extended_rdf' in input_features:
            if 'tar' in rdf_dir:
                all_rdf = rdf_read_tar(data, rdf_dir)
            else:
                all_rdf = rdf_read(data, rdf_dir)
        
            # make all the rdf same length for machine learning input
            all_rdf = rdf_trim(all_rdf, trim=trim)
            X_data = rdf_flatten(all_rdf)
        if 'shell_similarity' in input_features:
            all_shell_simi = shell_similarity_read(data, rdf_dir)
            if X_data is None:
                X_data = all_shell_simi
            else: 
                X_data = np.hstack((X_data, all_shell_simi))
        if 'fourier_space' in input_features:
            scatter_factors = rdf_read(data, '../fourier_space_0.1_normal/')
            if X_data is None:
                X_data = scatter_factors
            else: 
                X_data = np.hstack((X_data, scatter_factors))
        if 'lattice_abc' in input_features:
            all_lattice = batch_lattice(data, method='abc')
            if X_data is None:
                X_data = all_lattice
            else:
                X_data = np.hstack((X_data, all_lattice))
        if 'lattice_matrix' in input_features:
            all_lattice = batch_lattice(data, method='matrix')
            if X_data is None:
                X_data = all_lattice
            else:
                X_data = np.hstack((X_data, all_lattice))
        if 'distance_matrix' in input_features:
            assert(dist_matrix is not None)
            # Change to 2D version of train_test_split (also in sub-packages)
            train_test_split_old = train_test_split
            train_test_split = train_test_split_2D
            
            calc_obs_vs_pred = calc_obs_vs_pred_2D
            print('Reading distance matrix ... ', end=''),
            X_data = pd.read_csv(dist_matrix, index_col=0)
            print('Done')
        
    # the following line save X_data for check, but this is rarely used
    #np.savetxt(os.path.join(output_dir, 'X_data'), X_data, delimiter=' ',fmt='%.3f')

    # target_type can be continuous categorical ordinal
    # or multi-cont, multi-cate, multi-ord
    target_type = None

    # calculate the target values
    if target == 'bulk_modulus':
        target_type = 'continuous'
        y_data = np.array([ x['elasticity.K_VRH'] for x in data ])
        if args.log_target:
            y_data = np.log10(y_data)
    elif target == 'shear_modulus':
        target_type = 'continuous'
        y_data = np.array([ x['elasticity.G_VRH'] for x in data ])
        if args.log_target:
            y_data = np.log10(y_data)
    elif target == 'density':
        target_type = 'continuous'
        y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').density 
                            for x in data ])
    elif target == 'volume_per_atom':
        target_type = 'continuous'
        y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').volume 
                            / len(Structure.from_str(x['cif'], fmt='cif')) 
                            for x in data ])
    elif target == 'volume':
        target_type = 'continuous'
        y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').volume 
                            for x in data ])
    elif target == 'space_group_number':
        target_type = 'continuous'
        y_data = np.array([ Structure.from_str(x['cif'], fmt='cif').get_space_group_info()[1]
                            for x in data ])
    elif target == 'number_of_species':
        #target_type = 'ordinal'
        #y_data = np.array([ len(Structure.from_str(x['cif'], fmt='cif').symbol_set) 
        #                    for x in data ])
        target_type = 'categorical'
        y_data = [ str(len(Structure.from_str(x['cif'], fmt='cif').symbol_set)) 
                            for x in data ]
    elif target == 'number_of_atoms':
        target_type = 'ordinal'
        y_data = np.array([ len(Structure.from_str(x['cif'], fmt='cif')) 
                            for x in data ])
    elif target == 'composition':
        target_type = 'multi-cont'
        y_data, elem_symbols = composition_one_hot(data=data, method='percentage', index='pettifor')
    elif target == 'type_of_elements':
        target_type = 'multi-cate'
        y_data, elem_symbols = composition_one_hot(data=data, method='only_type', index='pettifor')
    elif target == 'bonding_type':
        target_type = 'multi-cate'
        y_data = bonding_matrix(data=data)
    elif target == 'average_coordination':
        target_type = 'continuous'
        y_data = np.array([ x['average_coordination'] for x in data ])
    elif target == 'average_bond_length':
        target_type = 'continuous'
        y_data = np.array([ x['average_bond_length'] for x in data ])
    elif target == 'bond_length_std':
        target_type = 'continuous'
        y_data = np.array([ x['bond_length_std'] for x in data ])
        if args.log_target:
            y_data = np.log10(y_data + 1)
    elif target == 'ave_bond_std':
        target_type = 'continuous'
        y_data = np.array([ x['ave_bond_std'] for x in data ])
    elif target == 'coord_num_std':
        target_type = 'continuous'
        y_data = np.array([ x['coord_num_std'] for x in data ])
    elif target == 'num_sg_operation':
        target_type = 'continuous'
        y_data = np.array([ x['num_sg_operation'] for x in data ])
    elif target == 'band_gap':
        assert 'band_gap' in data[0].keys()
        target_type = 'continuous'
        y_data = np.array([ x['band_gap'] for x in data ])
    else:
        print('This target is not support, please check help')
        exit()

    print('Setting up algorithm ... ', end='')
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
    elif funct_name == 'elastic_net':
        funct = ElasticNet(alpha=1.0, l1_ratio=0.7)
    elif funct_name == 'rf':
        if target_type in ('categorical', 'multi-cate'):
            funct = RandomForestClassifier(max_depth=2, random_state=0)
        else:
            funct = RandomForestRegressor(max_depth=2, random_state=0)
    elif funct_name == 'knn_reg':
        if dist_matrix is None:
            print('knn requires a pre-computed distance matrix')
            exit()
        funct = KNeighborsRegressor(n_neighbors=1, metric='precomputed')
    else:
        print('this algorithm is not supported, please check help')
        exit()

    print('Done')

    # use a 80/20 split except otherwise stated, e.g. in varying test_size
    test_size = 0.05
    if task == 'test_size_depend':
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
            elif target_type == 'categorical'and target == 'number_of_species':
                # now only implemented for number of species
                #y_pred[np.where(y_pred > 5)] = 5
                #y_pred = list(map(str, list(np.int64(y_pred + 0.5))))
                #print(type(y_test), type(y_test[0]), y_test[0], type(y_pred), type(y_pred[0]), y_pred[0])
                #pred_acc = metrics.accuracy_score(y_test, y_pred)
                y_test = list(map(float, y_test))
                pred_acc = metrics.mean_absolute_error(y_test, y_pred)
            elif target_type == 'multi-cate':
                if target == 'bonding_type':
                    y_test = bond_to_atom(y_test)
                    np.where(y_pred > 0.2, y_pred, 0)
                    y_pred = bond_to_atom(y_pred)

                if metrics_method == 'default':
                    pred_acc = [round(metrics.coverage_error(y_test, y_pred), 3),
                            round(metrics.label_ranking_average_precision_score(y_test, y_pred), 3),
                            round(metrics.label_ranking_loss(y_test, y_pred), 3) ]
                elif metrics_method == 'emd':
                    #for nelem in range(2,8):
                    y_bin = binarize_output(y_test, y_pred, threshold=None, 
                                            nelem=None, save_to_file=False)
                    pred_acc = emd_of_two_compositions(y_test, y_bin).mean()
                    #print(test_size, nelem, pred_acc)
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
    
    elif task == 'obs_vs_pred':
        calc_obs_vs_pred(funct=funct, X_data=X_data, y_data=y_data, test_size=test_size,
                        outdir= output_dir)
    
    elif task == 'confusion_matrix':
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
            np.savetxt(os.path.normpath(os.path.join(output_dir, 'confusion_matrix_' + str(round(test_size,3)))), 
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
            np.savetxt(os.path.normpath(os.path.join(output_dir, 'confusion_matrix_multiclass'), cm, delimiter=' ', fmt='%.3f'))
        else:
            print('This target does not support confusion matrix')
    
    elif task == 'grid_search':
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
    
    elif task == 'learning_curve':
        print('starting learning curve')
        calc_learning_curve(funct=funct, X_data=X_data, y_data=y_data, test_size=test_size)
        print('Done')

    elif task == 'random_guess':
        if target == 'number_of_species':
            nclass = 5
            max_class = 3

            y_pred_const = np.full(len(y_data), max_class)
            pred_acc_const = metrics.mean_absolute_error(y_data, y_pred_const)
            y_pred_rand = np.int64(np.random.rand(len(y_data)) * nclass + 1)
            pred_acc_rand = metrics.mean_absolute_error(y_data, y_pred_rand)

            print('constant {} ; random {}'.format(pred_acc_const, pred_acc_rand))
        
        elif target == 'type_of_elements':
            nelem = 78
            y_pred_rand = []
            for i in range(len(y_data)):
                y_rand = np.zeros(nelem)
                for j in np.random.randint(0, high=nelem, size=3):
                    y_rand[j] = 1
                y_pred_rand.append(y_rand)

            print(emd_of_two_compositions(y_data, y_pred_rand, pettifor_index=True).mean())

    elif task == 'emd_visual':
        X_train, X_test, y_train, y_test = \
                train_test_split(X_data, y_data, test_size=test_size, random_state=1)
        funct.fit(X_train, y_train)
        y_pred = funct.predict(X_test)
        
        for threshold in np.linspace(0.2, 0.6, 5):
            y_bin = binarize_output(y_test, y_pred, threshold=threshold, save_to_file=False)
            pred_acc = emd_of_two_compositions(y_test, y_bin)
            np.savetxt(os.path.normpath(os.path.join(output_dir, 'dist_histo_' + str(threshold))), pred_acc, delimiter=' ', fmt='%.3f')
            print(pred_acc.mean())
            
            large_samples, middle_samples, small_samples = \
                    n_best_middle_worst(y_test, y_bin, metrics_values=pred_acc, n_visual=100)
            np.savetxt(os.path.normpath(os.path.join(output_dir, 'large_sample_' + str(threshold))), large_samples,
                        delimiter=' ', fmt='%.3f')
            np.savetxt(os.path.normpath(os.path.join(output_dir, 'middle_sample_' + str(threshold))), middle_samples, 
                        delimiter=' ', fmt='%.3f')            
            np.savetxt(os.path.normpath(os.path.join(output_dir, 'small_sample_' + str(threshold))), small_samples, 
                        delimiter=' ', fmt='%.3f')

    else:
        print('This task is not supported')


