
'''
for rdf in smooth_rdf vanilla_rdf ; do cd $rdf ; for file in pero_distortion pero_lattice rp_srtio3 ; do python ../../../descriptors/data_explore.py --input_file  ../$file.json --task rdf_similarity  --output_file ../${file}_$rdf ; done ; cd .. ; done 
'''

import json
import math
import numpy as np
import pandas as pd
from collections import Counter
from pymatgen import Structure, Lattice


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


def normalize_fourier_space():
    '''
    '''
    for f in os.listdir('.'):
        fs = np.loadtxt(f, delimiter=' ')
        scale_factor = 100 / fs.max()
        np.savetxt('../fourier_space_0.1_normal/'+f, fs * scale_factor, delimiter=' ', fmt='%.3f')


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
