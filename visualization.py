'''
Visualization of the data
'''

import json
import heapq
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from pyemd import emd_with_flow


def calc_obs_vs_pred(funct, X_data, y_data, test_size, outdir='../'):
    '''
    The observation vs prediction plot
    '''
    X_train, X_test, y_train, y_test = \
            train_test_split(X_data, y_data, test_size=test_size, random_state=1)
    funct.fit(X_train, y_train)
    y_pred = funct.predict(X_test)
    y_pred_train = funct.predict(X_train)
    np.savetxt(outdir + 'test.' + str(test_size), 
                np.stack([y_test, y_pred]).transpose(), 
                delimiter=' ', fmt='%.3f')
    np.savetxt(outdir + 'train.' + str(test_size), 
                np.stack([y_train, y_pred_train]).transpose(), 
                delimiter=' ', fmt='%.3f')


def binarize_output(y_test, y_pred, threshold=None, nelem=None, save_to_file=False):
    '''
    Make the multilabe output decimals into 0 or 1
    according to the number of elements or a threshold

    Args:
        y_test: the ground truth label of the test data
        y_pred: prediciton of the test data
        threshold: above this value the output will be set to 1 and others 0
            If 'None', use the number (n) of non-zero labels in the ground
            truth, i.e. the top n largest values are set to 1 and others 0
        nelem: if None, use the number elements in the ground truth
            or given as an integer value
        save_to_file:
    Return:
        y_bin: a numpy array of binarized prediction
    '''
    y_bin = []
    nlabel = len(y_pred[0])
    for i, y in enumerate(y_pred):
        if threshold:
            y_new = np.zeros(nlabel)
            y_new[np.where(y > threshold)] = 1
        else:
            # the ground truth is 0 or 1, so the sum gives number of elements
            # note that sum in on the y_test i.e. ground truth
            if nelem:
                n_elem = nelem
            else:
                n_elem = y_test[i].sum()

            indice = y.argsort()[-n_elem:]
            y_new = np.zeros(nlabel)
            y_new[indice] = 1

        y_bin.append(y_new)

    if save_to_file:
        # save the ground truth of the test data
        # the current workdir is usually the one has the RDF files
        # so ../ means the files will be saved in parent folder
        np.savetxt('../multilabel_' + str(threshold) + '_test', 
                    y_test, delimiter=' ', fmt='%1i')
        # save the prediction values of the test data
        np.savetxt('../multilabel_' + str(threshold) + '_pred', 
                    y_pred, delimiter=' ', fmt='%.3f')
        # save the rounded values
        np.savetxt('../multilabel_' + str(threshold) + '_bin', 
                    np.stack(y_bin), delimiter=' ', fmt='%1i')

    return np.stack(y_bin)


def n_best_and_worst(y_test, y_pred, metrics_values, n_visual=100, 
                    method='confusion_matrix'):
    '''
    Output the n best and n worst prediction for visualization purpose
    This use heapq module to get n largest and smallest metrics values  
    For that has n middle values, see function 'n_best_middle_worst'

    Args:
        y_test: test data   
        y_pred: prediction data
        metrics_values: the criterion to select n best and worst
        n_visual: number of best and worst predictions 
        method: how the y vector is visualized
            original: original vectors, typically 78 elements
            confusion_matrix: 
    Return:

    '''
    # Give both index and value using heapq
    n_large = heapq.nlargest(n_visual, enumerate(metrics_values), key=lambda x: x[1])
    n_small = heapq.nsmallest(n_visual, enumerate(metrics_values), key=lambda x: x[1])

    large_samples = []
    small_samples = []
    middle_samples = []
    if method == 'original':
        for i, m_value in n_large:
            large_samples.append(y_test[i])
            large_samples.append(y_pred[i])
        for i, m_value in n_small:
            small_samples.append(y_test[i])
            small_samples.append(y_pred[i])
        return np.stack(large_samples).transpose(), np.stack(small_samples).transpose()
    elif method == 'confusion_matrix':
        for i, m_value in n_large:
            # confusion matrix, 0: true negetive,  1: false negative, 
            #                   2: false positive, 3: true positive
            large_samples.append(y_pred[i] * 2 + y_test[i])    
        for i, m_value in n_small:
            small_samples.append(y_pred[i] * 2 + y_test[i])
        return np.stack(large_samples).transpose(), np.stack(small_samples).transpose()
    else:
        print('This method is not supported in n_best_and_worst')


def n_best_middle_worst(y_test, y_pred, metrics_values, n_visual=100, 
                    method='confusion_matrix'):
    '''
    Output the n best, n middle, and n worst prediction for visualization purpose

    Args:
        y_test: test data   
        y_pred: prediction data
        metrics_values: the criterion to select n best and worst
        n_visual: number of best and worst predictions 
        method: how the y vector is visualized
            original: original vectors, typically 78 elements
            confusion_matrix: 
    Return:

    '''
    m_values = pd.DataFrame(metrics_values).sort_values(0)
    middle_index = int(len(metrics_values) / 2)
    middle_range = [ middle_index - int(n_visual / 2), middle_index + int(n_visual / 2)]

    large_samples = []
    small_samples = []
    middle_samples = []
    if method == 'original':
        for i in m_values[-n_visual:].index.values:
            large_samples.append(y_test[i])
            large_samples.append(y_pred[i])
        for i in m_values[middle_range[0]:middle_range[1]].index.values:
            middle_samples.append(y_test[i])
            middle_samples.append(y_pred[i])
        for i in m_values[:n_visual].index.values:
            small_samples.append(y_test[i])
            small_samples.append(y_pred[i])
        return np.stack(large_samples).transpose(), \
                np.stack(middle_samples).transpose(), \
                np.stack(small_samples).transpose()
    
    elif method == 'confusion_matrix':
        for i in m_values[-n_visual:].index.values:
            # confusion matrix, 0: true negetive,  1: false negative, 
            #                   2: false positive, 3: true positive
            large_samples.append(y_pred[i] * 2 + y_test[i])    
        for i in m_values[middle_range[0]:middle_range[1]].index.values:
            middle_samples.append(y_pred[i] * 2 + y_test[i])
        for i in m_values[:n_visual].index.values:
            small_samples.append(y_pred[i] * 2 + y_test[i])
        return np.stack(large_samples).transpose(), \
                np.stack(middle_samples).transpose(), \
                np.stack(small_samples).transpose()    

    else:
        print('This method is not supported in n_best_and_worst')


if __name__ == '__main__':
    pettifor = ['Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Yb', 'Eu', 'Y',  'Sc', 'Lu', 'Tm', 'Er', 'Ho', 
    'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 'Zr', 'Hf', 'Ti', 'Nb', 'Ta', 'V',  'Mo', 
    'W',  'Cr', 'Tc', 'Re', 'Mn', 'Fe', 'Os', 'Ru', 'Co', 'Ir', 'Rh', 'Ni', 'Pt', 'Pd', 'Au', 'Ag', 
    'Cu', 'Mg', 'Hg', 'Cd', 'Zn', 'Be', 'Tl', 'In', 'Al', 'Ga', 'Pb', 'Sn', 'Ge', 'Si', 'B',  'Bi', 
    'Sb', 'As', 'P',  'Te', 'Se', 'S', 'C', 'I', 'Br', 'Cl', 'N', 'O', 'F', 'H']

