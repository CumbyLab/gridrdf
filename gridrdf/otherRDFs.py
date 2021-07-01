'''
Other types of RDF as benchmark to the GRID.
'''


import numpy as np
import os
from pymatgen import Structure
try:
    from matminer.featurizers.structure import RadialDistributionFunction
    from matminer.featurizers.structure import PartialRadialDistributionFunction
except:
    print('matminer is not installed, cannot calculate original RDF')


def origin_rdf_histo(data, max_dist=10, bin_size=0.1, outdir='./'):
    '''
    Calcualte the vanilla RDF using matminer.  
    BEWARE! In the implementation of RDF in matminer, the endpoint
    is not included, e.g. with the default max_dist and bin_size, the  
    last x value is 9.9 instead of 10.0, this makes the list one item  
    shorter than the extend RDF.
    
    Args:
        data: input data from Materials Project
    Return:
        The RDFs are saved into files
    '''
    rdf_fn = RadialDistributionFunction(cutoff=max_dist, bin_size=bin_size)
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        rdf_bin = rdf_fn.featurize(struct)[0]['distribution']
        outfile = os.path.normpath(os.path.join(outdir, d['task_id']))
        np.savetxt(outfile, rdf_bin, delimiter=' ', fmt='%.3f')
    return


def partial_rdf(data, max_dist=10, bin_size=0.1):
    '''
    Partial RDF (implemented in matminer)

    Schütt, K. T., et al. (2014). 
    How to represent crystal structures for machine learning:  
    Towards fast prediction of electronic properties. 
    Physical Review B 89(20).205118
	
    Args:
        data: input data from Materials Project
    Return:
        The RDFs are saved into files
    '''
    prdf_fn = PartialRadialDistributionFunction(cutoff=max_dist, bin_size=bin_size)

    structs = []
    ids = []
    for d in data:
        structs.append(Structure.from_str(d['cif'], fmt='cif'))
        ids.append(d['task_id'])
    prdf_fn.fit(structs)

    for i, struct in enumerate(structs):
        prdf_bin = prdf_fn.featurize(struct)
        np.savetxt(ids[i], prdf_bin, delimiter=' ', fmt='%.3f')
    return


def mbtr_rdf():
    '''
    MBTR with k=2 (implemented in describ)

    Huo, H. and M. Rupp (2017) 
    Unified Representation of Molecules and Crystals for Machine Learning. 
    arXiv:1704.06439 

    Args:

    Return:

    '''
    pass


