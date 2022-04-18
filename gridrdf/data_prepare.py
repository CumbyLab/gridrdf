"""
Interface to collate or generate input data.

Helper functions to simplify extraction of data from the Materials Project
or to generate 'standard' datasets (such as artificial perovskites).
"""

# Bash script to generate RDF and GRID for structures
'''
for rdf in smooth_rdf vanilla_rdf ; do cd $rdf ; \
for file in pero_distortion pero_lattice rp_srtio3 ; \
do python ../../../descriptors/data_explore.py --input_file ../$file.json \
 --task rdf_similarity  --output_file ../${file}_$rdf ; \
done ; cd .. ; done 
'''

import json
import math
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from pymatgen import Structure, Lattice, MPRester

from .composition import elements_selection
from .data_io import rdf_read


def nacl():
    '''
    Generate NaCl structure, ususlly for test purpose

    Args:
        None
    Return:
        a pymatgen structure of NaCl
    '''
    nacl = Structure.from_spacegroup(
        'Fm-3m', Lattice.cubic(5.6),['Na', 'Cl'],[[0.5, 0.5, 0.5], [0, 0, 0]] )
    return nacl


def get_MP_bulk_modulus_data(APIkey): 
    '''
    Get all Material project structures with calculated elastic properties

    Args:
        APIkey: a string 'Hecxxxxxxxxxxx' needed to be applied from MP website
    Return:
        A json object containing
            MP task id
            Bulk modulus
            Shear modulus
            Elastic anisotropy
            CIF of structure
    '''
    # put personal API key here
    q = MPRester(APIKey)   
    d = q.query(criteria={'elasticity.K_VRH':{'$nin': [None, '']}}, 
                properties=['task_id', 
                            'elasticity.K_VRH', 
                            'elasticity.G_VRH', 
                            'elasticity.elastic_anisotropy', 
                            'cif'])
    return d


def get_ICSD_CIFs_from_MP(APIkey): 
    '''
    Get all ICSD structures in the Material project 

    Args:
        APIkey: a string 'Hecxxxxxxxxxxx' needed to be applied from MP website
    Return:
        A json object containing
            MP task id
            ICSD id
            CIF of structure
    '''
    q = MPRester(APIKey)   
    d = q.query(criteria={'icsd_ids':{'$nin': [None, []]}}, 
                properties=['task_id', 'icsd_ids', 'cif'])
    return d


def remove_nan():
    '''
    Remove NAN data from v6 by create a new list
    '''

    with open('MP_modulus_v8.json', 'r') as f:
        data = json.load(f)

    new_data = []
    for d in data:
        if not math.isnan(d['ave_bond_std']):
            new_data.append(d)

    with open('MP_modulus_v9.json', 'w') as f:
        json.dump(new_data, f, indent=1)


def insert_field(infile1='num_shell', infile2='MP_modulus_all.json', 
                outfile='MP_modulus_v4.json'):
    '''
    Insert new file in the json file

    Args:
        infile1: file containing values to be inserted, the field should
            consult data_explore.py
        infile2: file into which new field will be inserted
        outfile: a new file contains new inserted fields
    Return:
        None
    '''
    results = np.loadtxt(infile1, delimiter=' ')
    with open(infile2, 'r') as f:
        data = json.load(f)

    for i, d in enumerate(data):
        d['average_bond_length'] = results[i][2]
        d['bond_length_std'] = results[i][3]    

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=1)
    
    return


def json_order():
    '''
    Make the perovskite structure in the order of lattice constant
    '''

    with open('pero2.json','r') as f:
        data = json.load(f)

    l = []
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        l.append(struct.lattice.a)

    df2 = pd.DataFrame(l, columns=['lattice'])
    new_index = df2.sort_values('lattice').index.values

    d2 = []
    for i in new_index:
        d2.append(data[i])

    with open('pero3.json','w') as f:
        json.dump(d2,f,indent=1)

def make_distorted_perovskite():
    '''
    used for testing purpose for EMD + extended RDF as a similarity measure

    '''
    cif = "# generated using pymatgen\ndata_SrTiO3\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.94513000\n_cell_length_b   3.94513000\n_cell_length_c   3.94513000\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   90.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   SrTiO3\n_chemical_formula_sum   'Sr1 Ti1 O3'\n_cell_volume   61.40220340\n_cell_formula_units_Z   1\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Sr  Sr0  1  0.00000000  0.00000000  0.00000000  1\n  Ti  Ti1  1  0.50000000  0.50000000  0.50000000  1\n  O  O2  1  0.50000000  0.00000000  0.50000000  1\n  O  O3  1  0.50000000  0.50000000  0.00000000  1\n  O  O4  1  0.00000000  0.50000000  0.50000000  1\n"
    s = Structure.from_str(cif, fmt='cif')

    posits = [
            [0.50, 0.50, 0.50],
            [0.50, 0.50, 0.51],
            [0.50, 0.50, 0.52],
            [0.50, 0.50, 0.53],
            [0.50, 0.507, 0.507],
            [0.506, 0.506, 0.506]
        ]

    all_dict = []
    for i, posit in enumerate(posits):
        one_dict = {}
        one_dict['task_id'] = 'pero_distort_' + str(i)
        s[1] = 'Ti', posit
        one_dict['cif'] = s.to(fmt='cif')
        all_dict.append(one_dict)

    with open('pero_distortion.json','w') as f:
        json.dump(all_dict, f, indent=1)


def perovskite_different_lattice():
    cif = "# generated using pymatgen\ndata_SrTiO3\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.94513000\n_cell_length_b   3.94513000\n_cell_length_c   3.94513000\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   90.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   SrTiO3\n_chemical_formula_sum   'Sr1 Ti1 O3'\n_cell_volume   61.40220340\n_cell_formula_units_Z   1\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Sr  Sr0  1  0.00000000  0.00000000  0.00000000  1\n  Ti  Ti1  1  0.50000000  0.50000000  0.50000000  1\n  O  O2  1  0.50000000  0.00000000  0.50000000  1\n  O  O3  1  0.50000000  0.50000000  0.00000000  1\n  O  O4  1  0.00000000  0.50000000  0.50000000  1\n"
    s = Structure.from_str(cif, fmt='cif')

    all_dict = []
    for lat in np.linspace(3, 6, 61):
        one_dict = {}
        one_dict['task_id'] = 'pero_latt_' + str(round(lat,3))
        new_latt = Lattice.from_parameters(lat, lat, lat, 90, 90, 90)
        s.lattice = new_latt
        one_dict['cif'] = s.to(fmt='cif')
        all_dict.append(one_dict)

    with open('pero_lattice.json','w') as f:
        json.dump(all_dict, f, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset prepare',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', type=str, default='../MP_modulus.json',
                        help='the bulk modulus and structure from Materials Project')
    parser.add_argument('--output_file', type=str, default='subset.json',
                        help='outpu')
    parser.add_argument('--rdf_dir', type=str, default='./',
                        help='dir has all the rdf files')
    parser.add_argument('--task', type=str, default='',
                        help='what to do with the dataset: \n' +
                            '   subset_composition: select a subset which have specified elements: \n' +
                            '   subset_rdf_len: drop all the ext-RDF with a length less than 100 \n' +
                            '   subset_space_group: '
                      )
    parser.add_argument('--elem_list', type=str, default='O',
                        help='only used for subset task')

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    rdf_dir = args.rdf_dir
    task = args.task
    elem_list = args.elem_list

    with open(input_file,'r') as f:
        data = json.load(f)
    
    if task == 'subset_composition':
        print(elem_list)
        subset = elements_selection(data, elem_list=elem_list.split(), mode='consist')
        # note that 'data' is also changed because it is defined in __main__
        with open(output_file, 'w') as f:
            json.dump(subset, f, indent=1)

    if task == 'subset_rdf_len':
        all_rdf = rdf_read(data, rdf_dir)
        # note that 'data' is also changed because it is defined in __main__
        for i, d in enumerate(data[:]):
            if len(all_rdf[i]) < 100:
                data.remove(d)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=1)

    elif task == 'subset_space_group':
        sg_grouped_structs = {}
        for sg_num in range(230, 0, -1):
            sg_grouped_structs[sg_num] = []
    
        for d in data:
            struct = Structure.from_str(d['cif'], fmt='cif')
            sg_num = struct.get_space_group_info()[1]
            sg_grouped_structs[sg_num].append(d)

        for sg_num in range(230, 0, -1):
            with open(output_file + str(sg_num) + '.json', 'w') as f:
                json.dump(sg_grouped_structs[sg_num], f, indent=1)