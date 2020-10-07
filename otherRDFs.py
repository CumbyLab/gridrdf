'''
Other types of RDF as benchmark to the extend RDF
'''

import numpy as np
from pymatgen import Structure
from matminer.featurizers.structure import RadialDistributionFunction
from matminer.featurizers.utils.grdf import Histogram as PartialRDF


def origin_rdf_histo(data, max_dist=10, bin_size=0.1):
    '''
    Calcualte the vanilla RDF using matminer.  
    BEWARE! In the implementation of RDF in matminer, the endpoint
    is not included, e.g. with the default max_dist and bin_size, the  
    last x value is 9.9 instead of 10.0, this makes the list one item  
    shorter than the extend RDF.
    
    Args:
        data: input data from Materials Project
    Return:

    '''
    rdf_fn = RadialDistributionFunction(cutoff=max_dist, bin_size=bin_size)
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        rdf_bin = rdf_fn.featurize(struct)[0]['distribution']
        outfile = d['task_id']
        np.savetxt(outfile, rdf_bin, delimiter=' ', fmt='%.3f')
    return


def partial_rdf():
    '''
    Partial RDF (implemented in matminer)

    Schütt, K. T., et al. (2014). 
    How to represent crystal structures for machine learning:  
    Towards fast prediction of electronic properties. 
    Physical Review B 89(20).205118
	
    Args:

    Return:

    '''
    pass



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


