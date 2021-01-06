
import json
import math
import numpy as np
import pandas as pd
from collections import Counter
from pymatgen import Structure, Lattice
from pyemd import emd_with_flow

try:
    from ElMD import ElMD
except:
    print('Element-EMD module not installed')


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


def emd_example():
    '''
    Test the Earth Mover's Distance (EMD) using similarity matrix 
    against the EMD in the literature
    https://github.com/lrcfmd/ElMD/
    '''
    elem_emd = ElMD()
    comp1 = elem_emd._gen_vector('Li0.7Al0.3Ti1.7P3O12')
    comp2 = elem_emd._gen_vector('La0.57Li0.29TiO3')

    petiffor_emd = elem_emd._EMD(comp1, comp2)

    comp1_reindex = pd.DataFrame(comp1, index=modified_petiffor)
    comp1_reindex = comp1_reindex.reindex(index=petiffor)
    comp2_reindex = pd.DataFrame(comp2, index=modified_petiffor)
    comp2_reindex = comp2_reindex.reindex(index=petiffor)

    dist_matrix = pd.read_csv('similarity_matrix.csv', index_col='ionA').values
    dist_matrix = dist_matrix.copy(order='C')

    em = emd_with_flow(comp1_reindex.values[:,0], comp2_reindex.values[:,0], dist_matrix)
    simi_matrix_emd = em[0]
    emd_flow = pd.DataFrame(em[1], columns=petiffor, index=petiffor)
    emd_flow.replace(0, np.nan).to_csv('emd_flow.csv')


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


def analysis_emd_100():
    '''
    '''
    df = pd.DataFrame([], index=np.linspace(0.2, 0.6, 5))
    for i in ['small', 'middle', 'large']:
        for thresh in np.linspace(0.2, 0.6, 5):
            data = np.loadtxt(i + '_sample_' + str(thresh), delimiter=' ')
            for val in range(4):
                df.loc[thresh, i + str(val)] = np.count_nonzero(data == val)
    df.to_csv('analysis.csv')


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


def dist_matrix_1d(nbin=100):
    '''
    Generate distance matrix for earth mover's distance use
    '''
    dist_matrix = pd.DataFrame()
    for x in range(nbin):
        for y in range(nbin):
            dist_matrix.loc[x, y] = abs(x-y)
    return dist_matrix.values


if __name__ == '__main__':
    modified_petiffor = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 
                    'Fr', 'Cs', 'Rb', 'K', 'Na', 'Li', 'Ra', 'Ba', 'Sr', 'Ca', 
                    'Eu', 'Yb', 'Lu', 'Tm', 'Y', 'Er', 'Ho', 'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 
                    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 
                    'Sc', 'Zr', 'Hf', 'Ti', 'Ta', 'Nb', 'V', 'Cr', 'Mo', 'W', 'Re', 
                    'Tc', 'Os', 'Ru', 'Ir', 'Rh', 'Pt', 'Pd', 'Au', 'Ag', 'Cu', 
                    'Ni', 'Co', 'Fe', 'Mn', 'Mg', 'Zn', 'Cd', 'Hg', 
                    'Be', 'Al', 'Ga', 'In', 'Tl', 'Pb', 'Sn', 'Ge', 'Si', 'B', 'C', 
                    'N', 'P', 'As', 'Sb', 'Bi', 'Po', 'Te', 'Se', 'S', 'O', 'At', 'I', 'Br', 'Cl', 'F', 'H']
    petiffor = ['Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Yb', 'Eu', 'Y',  'Sc', 'Lu', 'Tm', 'Er', 'Ho', 
                'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 'Zr', 'Hf', 'Ti', 'Nb', 'Ta', 'V',  'Mo', 
                'W',  'Cr', 'Tc', 'Re', 'Mn', 'Fe', 'Os', 'Ru', 'Co', 'Ir', 'Rh', 'Ni', 'Pt', 'Pd', 'Au', 'Ag', 
                'Cu', 'Mg', 'Hg', 'Cd', 'Zn', 'Be', 'Tl', 'In', 'Al', 'Ga', 'Pb', 'Sn', 'Ge', 'Si', 'B',  'Bi', 
                'Sb', 'As', 'P',  'Te', 'Se', 'S', 'C', 'I', 'Br', 'Cl', 'N', 'O', 'F', 'H']
