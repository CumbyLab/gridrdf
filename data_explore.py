
import numpy as np
import json
import argparse
from pymatgen import Structure
try:
    from matminer.featurizers.structure import StructuralComplexity
except:
    print('matminer is not installed, cannot calculate structural complexity')

from extendRDF import rdf_histo, get_rdf_and_atoms


def struct_complex(data):
    '''
    Args:
        data: input data
    Return:
        number of atoms
        structural complexicity per atom and per cell
        bulk modulus
    '''
    sc = StructuralComplexity()

    result = []
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        complex_atom, complex_cell = sc.featurize(struct)
        result.append([ len(struct), complex_atom, complex_cell, 
                        d['elasticity.K_Voigt'] ])
    return np.array(result)


def batch_rdf(data, max_dist=10, normalize=True):
    '''
    Read structures and output the extend RDF

    Args:
        data: input data
        normalize: RDF dividing the number of atoms per unit cell
    Return:

    '''
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        prim_cell_list = list(range(len(struct)))
        rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
                                        max_dist=max_dist)
        rdf_bin = rdf_histo(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_size=0.1)
        if normalize:
            rdf_bin = rdf_bin / len(struct)
        outfile = d['task_id']
        np.savetxt(outfile, rdf_bin, delimiter=',', fmt='%i')
    return


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Data explore')
    parse.add_argument('--input', type=str, default='bulk_modulus_cif.json',
                        help='')
    parse.add_argument('--output', type=str, default='results',
                        help='Output results')

    args = parse.parse_args()
    infile = args.input
    outfile = args.output

    with open(infile,'r') as f:
        data = json.load(f)

    batch_rdf(data)
    #result = struct_complex(data=data)
    #np.savetxt(outfile, result, delimiter=' ',fmt='%.3f')