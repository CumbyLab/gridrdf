'''
Other types of RDF as benchmark to the extend RDF
'''

import numpy as np
from matminer.featurizers.structure import RadialDistributionFunction
from matminer.featurizers.utils.grdf import Histogram as PartialRDF


def origin_rdf_histo(structures, max_dist=10, bin_size=0.1):
    '''
    Calcualte the vanilla RDF using matminer.  
    BEWARE! In the implementation of RDF in matminer, the endpoint
    is not included, e.g. with the default max_dist and bin_size, the  
    last x value is 9.9 instead of 10.0, this makes the list one item  
    shorter than the extend RDF.
    
    Args:
        structures: a list of pymatgen structures
    Return:
        a ndarray with dimension [x,y], x is number of compounds i.e. 
        the length of the input list, y is the number of histogram 
        points of RDF
    '''
    rdf_fn = RadialDistributionFunction(cutoff=max_dist, bin_size=bin_size)
    results = rdf_fn.featurize_many(structures)

    rdfs = []
    for result in results:
        rdfs.append(result[0]['distribution'])
    return np.stack(rdfs)


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


