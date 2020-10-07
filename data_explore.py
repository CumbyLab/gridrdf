
import numpy as np
import json
import time
import gzip
import argparse
from pymatgen import Structure
try:
    from matminer.featurizers.structure import StructuralComplexity
except:
    print('matminer is not installed, cannot calculate structural complexity')

from extendRDF import rdf_histo, get_rdf_and_atoms
from otherRDFs import origin_rdf_histo


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


def batch_rdf(data, max_dist=10, normalize=True, gzip_file=False):
    '''
    Read structures and output the extend RDF

    Args:
        data: input data from Materials Project
        normalize: RDF dividing the number of atoms per unit cell
        gzip_file: in case RDF files take too much disk space, set 
                    this to true to gzip the files (not yet test)
    Return:

    '''
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        prim_cell_list = list(range(len(struct)))
        rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
                                        max_dist=max_dist)
        rdf_bin = rdf_histo(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_size=0.1)

        fmt = '%i'
        if normalize:
            rdf_bin = rdf_bin / len(struct)
            fmt = '%.3f'
        outfile = d['task_id']

        if gzip_file:
            with gzip.open(outfile+'.gz', 'w') as f:
                # not yet test, need test before use
                f.write(rdf_bin.tostring())
        else:
            np.savetxt(outfile, rdf_bin, delimiter=' ', fmt=fmt)
        time.sleep(0.1)
    return


def num_of_shells(data, dir):
    '''
    Calculate the number of nearest neighbor shells in RDF for each compound

    Args:
        data: the bulk modulus data and structure from Materials Project
        dir: the dir contains the calculated RDFs
    Return:
        a list of number of shells, bulk modulus and number of atoms
    '''
    results = []
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        num_shell = sum(1 for line in open(dir + '/' + d['task_id']))
        results.append(np.array([num_shell, d['elasticity.K_Voigt'], len(struct)]))
    return np.stack(results)


def rdf_value_stat(data, dir):
    '''
    Remove all the 0 values in all RDF and put all other values in one list.  
    To get a statistics intuition of the RDF values distribution histogram.

    Args:
        dir: the dir contains the calculated RDFs
    Return:
        a list of all non-zero RDF values
    '''
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        len(struct)
        outfile = d['task_id']
    return


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Data explore')
    parse.add_argument('--input', type=str, default='bulk_modulus_cif.json',
                        help='the bulk modulus and structure from Materials Project')
    parse.add_argument('--rdf_dir', type=str, default='./',
                        help='dir has all the rdf files')
    parse.add_argument('--output', type=str, default='results',
                        help='Output results')
    parse.add_argument('--max_dist', type=float, default=10.0,
                        help='Cutoff distance of the RDF')

    args = parse.parse_args()
    infile = args.input
    outfile = args.output
    max_dist = args.max_dist
    rdf_dir = args.rdf_dir

    with open(infile,'r') as f:
        data = json.load(f)

    results = num_of_shells(data=data, dir=rdf_dir)
    np.savetxt(outfile, results, delimiter=' ',fmt='%.3f')

    #origin_rdf_histo(data, max_dist=max_dist)
    #batch_rdf(data, max_dist=max_dist)
    #results = struct_complex(data=data)

