
import numpy as np
import gzip, tarfile
from tqdm import tqdm
import os


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
    for d in tqdm(data, desc='rdf read', mininterval=10):
        #rdf_file = os.path.normpath(os.path.join(rdf_dir, d['task_id']))
        #if zip_file:
        #    with gzip.open(rdf_file + '.gz', 'r') as f:
        #        rdf = np.loadtxt(f, delimiter=' ')
        #else:
        #    rdf = np.loadtxt(rdf_file, delimiter=' ')
        rdf = _rdf_single_read(d, rdf_dir, zip_file)
        all_rdf.append(rdf)
    return all_rdf
    
def _rdf_single_read(data_row, rdf_dir, zip_file=False):
    """ Read single data file """
    rdf_file = os.path.normpath(os.path.join(rdf_dir, data_row['task_id']))
    if zip_file:
        with gzip.open(rdf_file + '.gz', 'r') as f:
            rdf = np.loadtxt(f, delimiter=' ')
    else:
        rdf = np.loadtxt(rdf_file, delimiter=' ')
    return rdf

def _rdf_single_read_star(args):
    """ Make multiprocessing work with multiple arguments without starmap"""
    return _rdf_single_read(*args)
    
def rdf_read_parallel(data, rdf_dir, zip_file=False, procs=None):
    """ 
    Read rdf files in parallel using multiprocessing
    
    See rdf_read for (serial) documentation and usage
    
    """
    import multiprocessing as mp
    
    print("Reading data in parallel")
    
    pool = mp.Pool(procs)
    args = [(i, rdf_dir, zip_file) for i in data]
    rdf_collect = list(tqdm(pool.imap(_rdf_single_read_star, args), total=len(args)))
    
    pool.close()
    pool.join()
    
    print("Data read")
    
    return rdf_collect
    
    
    
    
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


