"""
Miscellaneous utility functions to handle structures and GRID descriptions.

"""

import numpy as np
import pandas as pd
import json
import time
import gzip
import argparse
import os
from tqdm import tqdm
from pyemd import emd, emd_with_flow
from pymatgen import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer 
try:
    from matminer.featurizers.structure import StructuralComplexity
except:
    print('matminer is not installed, cannot calculate structural complexity')

from .extendRDF import rdf_histo, rdf_kde, get_rdf_and_atoms, shell_similarity
from .composition import elements_count, bonding_type
from .otherRDFs import origin_rdf_histo


def batch_shell_similarity(all_rdf, method='append'):
    '''
    Calculate shell similarity for multiple RDFs
    
    Args:
        all_rdf: multiple RDFs with the same length
        method: how shell similarity is used in the X input
            append: append the similarity values after rdfs
            only: only use similarity values as the X input
    Return:
        RDF with shell similarity or similarity only
    '''
    rdf_shell_simi = []
    for rdf in all_rdf:
        if method == 'append':
            rdf_shell_simi.append(np.append(rdf, shell_similarity(rdf)))
        elif method == 'only':
            rdf_shell_simi.append(shell_similarity(rdf))
    return np.stack(rdf_shell_simi)


def rdf_trim(all_rdf, trim='minimum'):
    '''
    Trim the rdfs to the same length for kernel methods training

    Args:
        all_rdf: a list of np array (rdfs) with different length
        trim: must be one of 'no', 'minimum' or an integer
            no: no trim when the RDF already have the same length, 
            minimum: all rdf trim to the value of the smallest rdf
            integer value: if a value is given, all rdf longer than 
                this value will be trimmed, short than this value will  
                add 0.000 to the end
    Return:
        a two dimensional matrix, first dimension is the number of
        samples, and the second dimension is the flatten 1D rdf    
    '''
    rdf_lens = []
    for rdf in all_rdf:
        rdf_lens.append(len(rdf))

    if trim == 'minimum':
        min_len = np.array(rdf_lens).min()
        all_rdf = [ x[:min_len] for x in all_rdf]
    elif isinstance(trim, int):
        for i, rdf_len in enumerate(rdf_lens):
            if rdf_len < trim:
                b = trim - rdf_len
                if b != 0:
                    if len(all_rdf[0].shape) == 2:
                        nbins = len(all_rdf[i][0])
                        all_rdf[i] = np.append( all_rdf[i], [[0.0] * nbins] * b, axis=0 )
                    elif len(all_rdf[0].shape) == 1:
                        all_rdf[i] = np.append( all_rdf[i], [0.0] * b )
            else:
                all_rdf[i] = all_rdf[i][:trim]

#            print(len(all_rdf[i]))
    elif trim == 'none':
        pass 
    else:
        print('wrong value provided for trim') 

    return np.stack(all_rdf)


def rdf_flatten(all_rdf):
    '''
    If the rdf is not 1d, make it 1d for machine learning input
    This is done after trim so np.stack can be used

    Args:
        all_rdf: a 2D or 3D np array (rdfs) with the same length, 
            with the first dimension be number of samples
    Return:
        a 2D np array of rdfs
    '''
    if len(all_rdf[0].shape) == 2:
        all_rdf = [ x.flatten() for x in all_rdf]
    return np.stack(all_rdf)


def batch_lattice(data, method='matrix'):
    '''
    Calculate lattice parameters of each structure

    Args:
        data: input data from Materials Project
        method:
            abc: binned rdf, equal to the 'uniform' method in kde,  
                note that the 
            matrix: use kernel density estimation to give a smooth curve
    Return:
        lattice parameters
    '''
    all_lattice = []
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        if method == 'matrix':
            all_lattice.append(struct.lattice.matrix.reshape(9))
        elif method == 'abc':
            all_lattice.append(np.array(struct.lattice.abc + struct.lattice.angles))
    return np.stack(all_lattice)


def bond_length_statis(structure): 
    '''
    Calculated some bond length statistics values in the structure
    the nearest atoms pairs are determined using pymatgen CrystalNN module

    Currently the range of bond length has some problem for structure with
    no bonding, i.e. empty np array

    Args:
        structrue: pymatgen structure
    Return:
        the mean value and standard deviation and all the cation-anion
        bond length in the crystal structure
    '''
    nn = CrystalNN()
    bond_len = []
    for j in range(len(structure)): 
        for i in nn.get_nn_info(structure, j):       
            bond_len.append(structure[j].distance(structure[i['site_index']]))
    bond_len = np.array(bond_len)
    return bond_len.mean(), bond_len.std()#, bond_len.ptp()


def average_coordination(structure):
    '''
    Calculation of average coordination number over every site in a structure
    using Vorini method

    Args:
        structure: pymatgen structure
    Return:
        Average coordination number
    '''
    nn = CrystalNN()
    ave_coord_num = []
    for atom_site in range(len(structure)):
        ave_coord_num.append(len(nn.get_nn_info(structure, atom_site)))
    return np.array(ave_coord_num).mean()


def bond_stat_per_site(structure):
    '''
    Get the bond length standard deviation of each site in the unit cell, the average
    Get the coordination number of each site, then standard deviation

    Args:
        structure: pymatgen structure
    Return:
        Average of bond length std of each site
        Standard deviation of coordination number
    '''
    nn = CrystalNN()
    coord_num = []
    bond_len_std = []
    for atom_site in range(len(structure)):
        struct_nn = nn.get_nn_info(structure, atom_site)
        coord_num.append(len(struct_nn))
        bond_len = []
        for i in struct_nn:       
            bond_len.append(structure[atom_site].distance(structure[i['site_index']]))
        bond_len_std.append(np.array(bond_len).std())
    return round(np.array(bond_len_std).mean(), 3), \
            round(np.array(coord_num).std(), 3)


def num_of_shells(data, dir):
    '''
    Calculate the number of nearest neighbor shells in RDF for each compound,  
    some other properties are also given for data analysis purpose

    Args:
        data: the bulk modulus data and structure from Materials Project
        dir: the dir contains the calculated RDFs
    Return:
        a list of properties, will change according to requrest
    '''
    #sc = StructuralComplexity()
    results = []
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        #complex_atom, complex_cell = sc.featurize(struct)
        '''num_shell = sum(1 for line in open(dir + '/' + d['task_id']))

        with open(dir + '/' + d['task_id']) as f:
            rdf = np.loadtxt(f, delimiter=' ')
            num_rdf_bar = np.count_nonzero(rdf[:30])
        '''
        bond_mean, bond_std = bond_length_statis(struct)

        results.append(np.array([
            #d['elasticity.K_Voigt'],           # bulk modulus
            #num_shell,                          # number of RDF shells
            #num_rdf_bar,                        # non-zeors in the first 30 RDF shell
            #struct.volume/len(struct),          # volume per atom
            struct.volume,                      # volume
            struct.get_space_group_info()[1],   # space group number
            bond_mean,
            bond_std,
            #,     # average coordination nunmber
            #struct.density,                     # density in g/cm^3
            len(struct),                        # number of atoms
            len(struct.symbol_set),             # number of types of elements
            #sc.featurize(struct)[0],    # structural complexcity per atom
        ]))

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
    parser = argparse.ArgumentParser(description='Data explore',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', type=str, default='../MP_modulus.json',
                        help='the bulk modulus and structure from Materials Project')
    parser.add_argument('--output_file', type=str, default='rdf_similarity',
                        help='currently for rdf similarity')
    parser.add_argument('--rdf_dir', type=str, default='./',
                        help='dir has all the rdf files')
    parser.add_argument('--task', type=str, default='num_shell',
                        help='which property to be calculated: \n' +
                            '   num_shell: number of RDF shells and other propreties \n' +
                            '   extend_rdf_bin: binned extend RDF of all the CIFs \n' +
                            '   extend_rdf_kde: kernel density estimation RDF \n' + 
                            '   origin_rdf: calcualte vanilla RDF of all the CIFs \n' + 
                            '   composition: element-wise statistics of all compositions \n' +
                            '   bonding_type: \n' +
                            '   shell_similarity: \n' +
                            '   '
                      )


    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    rdf_dir = args.rdf_dir
    task = args.task

    max_dist = args.max_dist

    with open(input_file,'r') as f:
        data = json.load(f)

    if task == 'num_shell':
        results = num_of_shells(data=data, dir=rdf_dir)
        outfile = '../num_shell'
        np.savetxt(outfile, results, delimiter=' ', fmt='%.3f')

    elif task == 'composition':
        elements_count(data)


    elif task == 'bonding_type':
        for d in data:
            struct = Structure.from_str(d['cif'], fmt='cif')
            d['bond_elem_list'], d['bond_num_list'] = bonding_type(struct)
        with open(input_file.replace('.json','_with_bond.json'), 'w') as f:
            json.dump(data, f, indent=1)

    elif task == 'bond_stat_per_site':
        for d in data:
            struct = Structure.from_str(d['cif'], fmt='cif')
            d['ave_bond_std'], d['coord_num_std'] = bond_stat_per_site(struct)
            d['num_sg_operation'] = len(SpacegroupAnalyzer(struct).get_space_group_operations())
        with open(input_file.replace('v7','v8'), 'w') as f:
            json.dump(data, f, indent=1)
            
    elif task == 'shell_similarity':
        for i, d in enumerate(data):
            if ( i % 100 ) == 0:
                print(i) 
            with open(rdf_dir + '/' + d['task_id']) as f:
                rdf = np.loadtxt(f, delimiter=' ')
                np.savetxt('../shell_similarity/' + d['task_id'], 
                            shell_similarity(rdf[:30]), 
                            delimiter=' ', fmt='%.3f')


    else:
        print('This task is not supported')

