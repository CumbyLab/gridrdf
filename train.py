

import os
import json
import argparse
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn import metrics


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
    parse = argparse.ArgumentParser(description='Train machine learning algorithm')
    parse.add_argument('--xdir', type=str, default='./',
                        help='All the rdf files for training')
    parse.add_argument('--yfile', type=str, default='../bm.json',
                        help='bulk modulus values as target')

    args = parse.parse_args()
    x_dir = args.xdir
    y_file = args.yfile

    # prepare the 
    all_rdf = read_and_trim_rdf(x_dir)
    with open (y_file,'r') as f:
        d = json.load(f)
    bm = np.array([ x['elasticity.K_Voigt'] for x in d ])
    x_train, x_test, y_train, y_test = train_test_split(all_rdf, bm, 
                                        test_size=0.3, random_state=109) 

    clf = KernelRidge(alpha=1.0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    metrics.accuracy_score(y_test, y_pred)
    #metrics.precision_score(y_test, y_pred)
    #metrics.recall_score(y_test, y_pred)