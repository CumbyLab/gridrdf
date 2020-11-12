'''
Visualization of the data
'''

import numpy as np


def calc_obs_vs_pred(funct, X_data, y_data, test_size):
    '''
    The observation vs prediction plot
    '''
    X_train, X_test, y_train, y_test = \
            train_test_split(X_data, y_data, test_size=test_size, random_state=1)
    funct.fit(X_train, y_train)
    y_pred = funct.predict(X_test)
    y_pred_train = funct.predict(X_train)
    np.savetxt('../test.' + str(test_size), 
                np.stack([y_test, y_pred]).transpose(), 
                delimiter=' ', fmt='%.3f')
    np.savetxt('../train.' + str(test_size), 
                np.stack([y_train, y_pred_train]).transpose(), 
                delimiter=' ', fmt='%.3f')


def binarize_output(y_test, y_pred, threshold=None, save_to_file=False):
    '''
    Make the multilabe output decimals into 0 or 1
    according to the number of elements or a threshold

    Args:
        y_test: the ground truth label of the test data
        y_pred: prediciton of the test data
        threshold: above this value the output will be set to 1 and others 0
            If 'None', use the number (n) of non-zero labels in the ground
            truth, i.e. the top n largest values are set to 1 and others 0
        save_to_file:
    Return:
        y_bin: a numpy array of binarized prediction
    '''
    y_bin = []
    nlabel = len(y_pred[0])
    for i, y in enumerate(y_pred):
        if threshold:
            n_elem = len(np.where(y > threshold))
        else:
            # thr ground truth is 0 or 1, so the sum gives number of elements
            # note that sum in on the y_test i.e. ground truth
            n_elem = y_test[i].sum()
        
        indice = y.argsort()[-n_elem:]
        y_new = np.zeros(nlabel)
        y_new[indice] = 1
        y_bin.append(y_new)

    if save_to_file:
        # save the ground truth of the test data
        # the current workdir is usually the one has the RDF files
        # so ../ means the files will be saved in parent folder
        np.savetxt('../multilabel_' + str(round(test_size,3)) + '_test', 
                    y_test, delimiter=' ', fmt='%1i')
        # save the prediction values of the test data
        np.savetxt('../multilabel_' + str(round(test_size,3)) + '_pred', 
                    y_pred, delimiter=' ', fmt='%.3f')
        # save the rounded values
        np.savetxt('../multilabel_' + str(round(test_size,3)) + '_bin', 
                    np.stack(y_bin), delimiter=' ', fmt='%1i')

    return np.stack(y_bin)


if __name__ == '__main__':
    pass

