
import numpy as np
import gzip, tarfile
from tqdm import tqdm


def rdf_read(data, rdf_dir, zip_file=False):
    '''
    Read the rdf files from the dir.  

    Args:
        data: modulus data from materials project
        rdf_dir: the dir has all (and only) the rdf files
        zip_file: if the RDFs are gzip file
    Return:
        a list of np array (rdfs) with different length, 
        first dimension is the number of
        samples, and the second dimension is the flatten 1D rdf
    '''
    all_rdf = []
    for d in tqdm(data, desc='rdf read', mininterval=60):
        rdf_file = rdf_dir + '/' + d['task_id']
        if zip_file:
            with gzip.open(rdf_file + '.gz', 'r') as f:
                rdf = np.loadtxt(f, delimiter=' ')
        else:
            rdf = np.loadtxt(rdf_file, delimiter=' ')
        all_rdf.append(rdf)
    return all_rdf


def shell_similarity_read(data, rdf_dir):
    '''
    Read the rdf files from the dir.  

    Args:
        data: modulus data from materials project
        rdf_dir: the dir has all (and only) the rdf files
    Return:

    '''
    all_shell_simi = []
    for d in data:
        shell_simi_file = rdf_dir + '/../shell_similarity/' + d['task_id']
        shell_simi = np.loadtxt(shell_simi_file, delimiter=' ')
        all_shell_simi.append(shell_simi.flatten())
    return np.stack(all_shell_simi)


def rdf_read_tar(data, x_file):
    '''
    Read the rdf files from a tar file.  

    Args:
        data: modulus data from materials project
        x_file: the tar file has rdfs
    Return:
        a list of np array (rdfs) with different length, 
        first dimension is the number of
        samples, and the second dimension is the flatten 1D rdf
    '''
    all_rdf = []
    with tarfile.open(x_file, 'r:*') as tar:
        for d in data:
            rdf_file = d['task_id']
            rdf = np.loadtxt(tar.extractfile(rdf_file), delimiter=' ')
            all_rdf.append(rdf)
    return all_rdf


def read_all_fs():
    '''
    '''
    import os
    import numpy as np
    import pandas as pd

    fs = []
    for f in os.listdir('.'):
        fs.append(np.loadtxt(f, delimiter=' '))

    df = pd.DataFrame(fs).transpose()


