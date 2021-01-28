
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from pymatgen.core.periodic_table import Element
from collections import Counter
from itertools import combinations
from pymatgen import Structure
from pymatgen.analysis.local_env import CrystalNN


def composition_one_hot(data, only_type=False, normalize=True, pettifor_index=True):
    '''
    Make the composition to fix size vector like one hot array
    !!!REINDEX FOR ONLY_TYPE=FASLE NOT IMPLEMENT!!!

    Args:
        data: the bulk modulus data and structure from Materials Project
        only_type: if true, only the types are included but not  
            the whole formula
        normalize: if normalize, the number of atoms in the output will 
            be given as percentage
        pettifor_index: whether transform the element order from the peroidic table
            number to Pettifor number
    Return:
        1. one hot reprentation in atomic array
        2. the order of the elements in the list
    '''
    pero_tab_nums = []
    if only_type:
        mlb = MultiLabelBinarizer()
        for d in data:
            struct = Structure.from_str(d['cif'], fmt='cif')
            species = struct.types_of_specie
            pero_tab_nums.append([ x.number for x in species ])
        #
        data_np = mlb.fit_transform(pero_tab_nums)
        elem_numbers = mlb.classes_.tolist()
        elem_symbols = [ Element.from_Z(x).name for x in elem_numbers ]
        if pettifor_index:
            Pettifor = ['Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Yb', 'Eu', 'Y',  'Sc', 'Lu', 'Tm', 'Er', 'Ho', 
                'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 'Zr', 'Hf', 'Ti', 'Nb', 'Ta', 'V',  'Mo', 
                'W',  'Cr', 'Tc', 'Re', 'Mn', 'Fe', 'Os', 'Ru', 'Co', 'Ir', 'Rh', 'Ni', 'Pt', 'Pd', 'Au', 'Ag', 
                'Cu', 'Mg', 'Hg', 'Cd', 'Zn', 'Be', 'Tl', 'In', 'Al', 'Ga', 'Pb', 'Sn', 'Ge', 'Si', 'B',  'Bi', 
                'Sb', 'As', 'P',  'Te', 'Se', 'S', 'C', 'I', 'Br', 'Cl', 'N', 'O', 'F', 'H']
            data_pd = pd.DataFrame(data_np, columns=elem_symbols)
            data_pd = data_pd.reindex(columns=Pettifor)
            data_np = data_pd.dropna().values
        return data_np, elem_symbols
    else:
        for d in data:
            struct = Structure.from_str(d['cif'], fmt='cif')
            species = struct.species
            pero_tab_nums.append([ x.number for x in species ])
        # use the counter method to one-hot the element numbers
        data_pd = pd.DataFrame([Counter(x) for x in pero_tab_nums])
        data_np = data_pd.fillna(0).sort_index(axis=1).to_numpy()
        if normalize:
            data_np = data_np/data_np.sum(axis=1, keepdims=True)

        # get the order of elements in the list and element symbols
        elem_numbers = data_pd.sort_index(axis=1).columns.tolist()
        elem_symbols = [ Element.from_Z(x).name for x in elem_numbers ]
        return np.ndarray.round(data_np, 3), elem_symbols


def elements_count(data):
    '''
    Count the elements distribution histogram in the compounds dataset,  
    and the covariance matrix of each pair of elements

    Args:
        data: a list of dicts with CIFs
    Return:
        write element histogram in elem_histo file
        write element-wise covariance matrix in elem_matrix file
    '''
    # This is the periodic table which DFT pesudopotentials are avaiable
    periodic_table = [
        'H',  'He', 
        'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne', 
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 
        'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
        'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe', 
        'Cs', 'Ba', 
                    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
                          'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 
                    'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu']
    # initialize two dicts
    elem_histo = {}
    elem_matrix = {}
    for elem in periodic_table:
        elem_histo[elem] = 0
        elem_matrix[elem] = {}
        for elem2 in periodic_table:
            elem_matrix[elem][elem2] = 0

    for d in data:
        elements = Structure.from_str(d['cif'], fmt='cif').symbol_set
        for elem in elements:
            elem_histo[elem] += 1
        for elem_pair in combinations(elements, 2):
            elem_matrix[elem_pair[0]][elem_pair[1]] += 1

    for elem in periodic_table:
        for elem2 in periodic_table:
            if elem_matrix[elem][elem2] == 0 and elem != elem2:
                elem_matrix[elem][elem2] = elem_matrix[elem2][elem]

    elem_histo = {elem:count for elem,count in elem_histo.items() if count != 0}
    elem_histo = DataFrame([elem_histo])
    elem_histo = elem_histo[(elem_histo != 0)]
    elem_histo.T.to_csv('elem_histo', sep=' ')

    #
    elem_matrix = DataFrame.from_dict(elem_matrix)
    # remove columns and rows with all zero
    elem_matrix = elem_matrix.loc[:, (elem_matrix != 0).any(axis=0)] # column
    elem_matrix = elem_matrix[(elem_matrix != 0).any(axis=1)]   # row
    elem_matrix.to_csv('elem_matrix', sep=' ')

    return


def elements_selection(data, elem_list, mode='include'):
    '''
    Select a subset contains or not contain certain elements.

    Args:
        data: a list of dicts with CIFs
        elem_list: a list of the elements of interest or no-interest
        mode:
            include: select structures have elements in elem_list
            exclude: drop structures have elements in elem_list
            consist: select structures made up of elements in elem_list
    Return:
        A new dataset after selection
    '''
    for d in data[:]:
        elements = Structure.from_str(d['cif'], fmt='cif').symbol_set
        if mode == 'include':
            if set(elem_list).isdisjoint(elements):
                data.remove(d)
        elif mode == 'exclude':
            if not set(elem_list).isdisjoint(elements):
                data.remove(d)
        elif mode == 'consist':
            if not set(elements).issubset(elem_list):
                data.remove(d)
    return data


def similarity_matrix(input_file='dist_matrix', normalize='inverse', order='pt_number'):
    '''
    Read the original file from the below reference, and convert into a dictionary

    Hautier, G., et al. (2011). 
    "Data mined ionic substitutions for the discovery of new compounds." 
    Inorganic Chemistry 50(2): 656-663.

    Args:
        input_file: the preprocessed supporting txt file (from the above paper)
                using the shell script below
        normalize: method for value normalization
            bound: all value divide by 20 (based on the maximum value)
            log: all value got log10
            inverse: 1 / values
        order: the order of element in the matrix
            pt_number: the order in periodic table, i.e. atomic number
                    typically for calculation purpose
            pettifor: typically for visualization purpose, following 
                    Pettifor, D. G. (1990). "Structure maps in alloy design." 
                    Journal of the Chemical Society, Faraday Transactions 86(8): 1209.
     Return:
        a pandas dataframe of the similarity matrix
    ====================================================
    #!/bin/sh
    outfile=dist_matrix
    > $outfile
    for i in `seq 78` # up to Bi and remove nobel gas
    do
        e1=`sed -n "$i p" mendlev` # element symbols in Pettifor order
        p1=`sed -n "$i p" Pettifor` # element symbol + valency in Pettifor order
        for j in `seq 78`
        do
            e2=`sed -n "$j p" mendlev`
            p2=`sed -n "$j p" Pettifor`
            if [ $i -gt $j ]
            then
                r=`grep $p1 ic102031h_si_001.txt | grep $p2`
                if [ -z "$r" ]
                then
                    grep -w $e1 ic102031h_si_001.txt | grep -w $e2 | head -n 1 >> $outfile
                else
                    echo $r >> $outfile
                fi
            fi
        done
    done
    sed -i 's/:/ /g' $outfile # make the valency a seperate column
    =========================================================
    '''
    # note that the index_col is after the use of selection
    d = pd.read_csv(input_file, sep=' ', index_col=[0,1], usecols=[0,2,4])
    # make it a two dimensional matrix
    d = d.unstack()
    # drop the multilevel when ustack
    d.columns = d.columns.droplevel()

    if order == 'pt_number':
        index = ['H', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F', 'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 
                'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
                'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
    # Pettifor order of elements
    elif order == 'pettifor':
        index = ['Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Yb', 'Eu', 'Y',  'Sc', 'Lu', 'Tm', 'Er', 'Ho', 
                'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 'Zr', 'Hf', 'Ti', 'Nb', 'Ta', 'V',  'Mo', 
                'W',  'Cr', 'Tc', 'Re', 'Mn', 'Fe', 'Os', 'Ru', 'Co', 'Ir', 'Rh', 'Ni', 'Pt', 'Pd', 'Au', 'Ag', 
                'Cu', 'Mg', 'Hg', 'Cd', 'Zn', 'Be', 'Tl', 'In', 'Al', 'Ga', 'Pb', 'Sn', 'Ge', 'Si', 'B',  'Bi', 
                'Sb', 'As', 'P',  'Te', 'Se', 'S', 'C', 'I', 'Br', 'Cl', 'N', 'O', 'F', 'H']
    # reindex, i.e. change the order of column and rows to Pettifor order  
    d = d.fillna(0)
    d = d.reindex(columns=index, index=index, fill_value=0) 
    d = d + d.transpose()
    # the maximum similarity number is 18.6, set the same element to 20, a bit arbitary
    np.fill_diagonal(d.values, 20)
    # then fill the zeros
    d.loc[:,'Pm'] = d.loc[:,'Sm'] # column, same as:  d['Pm'] = d['Sm']
    d.loc['Pm',:] = d.loc['Sm',:] # row
    # other zero mean very dissimilar, so set to a very small value
    d.replace(0, 0.1, inplace=True)

    if normalize == 'bound':
        d = d / 20
    elif normalize == 'log':
        d = np.log10(d)
    elif normalize == 'inverse':
        d = 1 / d
        np.fill_diagonal(d.values, 0)
    else:
        print('normalization method not supported')

    return d


def bonding_type(structure):
    '''
    Calculate bonding in a given structure.

    Args:
        structure: a pymatgen structure
    Return:
        A list of atomic pairs which form bonding
    '''
    nn = CrystalNN()
    bond_elem_list = []
    bond_num_list = []
    for i in list(range(len(struct))):
        site1 = struct[i].species_string
        num1 = struct[i].specie.number
        for neigh in nn.get_nn_info(struct, i):
            bond_elem_list.append(' '.join(sorted([site1, neigh['site'].species_string])))
            bond_num_list.append(' '.join(list(map(str, sorted([num1, neigh['site'].specie.number])))))

    bond_elem_list = list(set(bond_elem_list))
    bond_num_list = list(set(bond_num_list))
   
    return bond_elem_list, bond_num_list
            

def bonding_matrix(data):
    '''
    Convert the atomic pair of bonding into a matrix representation
    Remove the elements that don't exsit, then flatten it.

    Args:
        data: a list of dicts with CIFs
    Return:
        A 2D vector, first dimension is number of samples.
    '''
    periodic_table = ['H', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F', 'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 
            'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
            'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
            'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']

    all_bond_matrix = []
    num_elements = len(periodic_table)
    zeor_bond_matrix = pd.DataFrame(np.zeros((num_elements, num_elements)), 
                                    index=periodic_table, columns=periodic_table)
    for d in data:
        bond_matrix = zeor_bond_matrix
        bond_list = d['bond_elem_list']
        for bond in bond_list:
            elem1, elem2 = bond.split()
            bond_matrix.loc[elem1, elem2] = 1
            bond_matrix.loc[elem2, elem1] = 1
        all_bond_matrix.append(bond_matrix.values.flatten())
    # HERE NEED TO DELETE ZERO ROW/COLUMN AND THEN FLATTEN
    return np.stack(all_bond_matrix)

