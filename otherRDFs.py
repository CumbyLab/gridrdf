'''
Other types of RDF as benchmark to the extend RDF
'''


from matminer.featurizers.structure import RadialDistributionFunction
from matminer.featurizers.utils.grdf import Histogram as PartialRDF


def origin_rdf_histo(structure, max_dist=10, bin_size=0.1):
    '''
    Calcualte the vanilla RDF using matminer.  
    BEWARE! In the implementation of RDF in matminer, the endpoint
    is not included, e.g. with the default max_dist and bin_size, the  
    last x value is 9.9 instead of 10.0, this makes the list one item  
    shorter than the extend RDF.
    
    Args:
        struct: pymatgen structure
    Return:
        a nd array containing histogram of RDF
    '''
    rdf_fn = RadialDistributionFunction(cutoff=max_dist, bin_size=bin_size)  
    RDFs = rdf_fn.featurize(structure)  
    return RDFs[1]


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


