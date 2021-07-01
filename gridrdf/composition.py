""" Module to handle composition manipulations and analyses.

"""

import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from pymatgen.core.periodic_table import Element
from collections import Counter
from itertools import combinations
from pymatgen import Structure
from pymatgen.analysis.local_env import CrystalNN


def element_indice():
    '''
    type of element indice used 
    '''
    global modified_pettifor, pettifor, periodic_table, periodic_table_78

    # 78 elements without nobal gas and rare earch Ac
    # Put the elements with similar propeties together to give a better representation, see
    # Pettifor, D. G. (1990). "Structure maps in alloy design." 
    # Journal of the Chemical Society, Faraday Transactions 86(8): 1209.
    pettifor = [
        'Cs', 'Rb', 'K', 'Na', 'Li', 
        'Ba', 'Sr', 'Ca', 
        'Yb', 'Eu', 'Y',  'Sc', 'Lu', 'Tm', 'Er', 'Ho', 
        'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 
        'Zr', 'Hf', 'Ti', 'Nb', 'Ta', 'V',  'Mo', 'W',  'Cr', 'Tc', 'Re', 
        'Mn', 'Fe', 'Os', 'Ru', 'Co', 'Ir', 'Rh', 'Ni', 'Pt', 'Pd', 'Au', 'Ag', 'Cu', 
        'Mg', 'Hg', 'Cd', 'Zn', 'Be', 'Tl', 'In', 'Al', 'Ga', 'Pb', 'Sn', 'Ge', 'Si', 'B',  
        'Bi', 'Sb', 'As', 'P',  'Te', 'Se', 'S', 'C', 'I', 'Br', 'Cl', 'N', 'O', 'F', 'H'
    ]

    # 103 elements, A modified version of Pettifor, see
    # Glawe, H., et al. (2016). New Journal of Physics 18(9): 093011.
    # "The optimal one dimensional periodic table: a modified Pettifor chemical scale from data mining." 
    modified_pettifor = [
        'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 
        'Fr', 'Cs', 'Rb', 'K', 'Na', 'Li', 'Ra', 'Ba', 'Sr', 'Ca', 
        'Eu', 'Yb', 'Lu', 'Tm', 'Y', 'Er', 'Ho', 'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 
        'Sc', 'Zr', 'Hf', 'Ti', 'Ta', 'Nb', 'V', 'Cr', 'Mo', 'W', 'Re', 
        'Tc', 'Os', 'Ru', 'Ir', 'Rh', 'Pt', 'Pd', 'Au', 'Ag', 'Cu', 
        'Ni', 'Co', 'Fe', 'Mn', 'Mg', 'Zn', 'Cd', 'Hg', 
        'Be', 'Al', 'Ga', 'In', 'Tl', 'Pb', 'Sn', 'Ge', 'Si', 'B', 'C', 
        'N', 'P', 'As', 'Sb', 'Bi', 'Po', 'Te', 'Se', 'S', 'O', 'At', 'I', 'Br', 'Cl', 'F', 'H'
    ]

    # 89 elements in the order of atomic Z number
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
    ]

    # 78 elements version by removing nobal gas and six rare-earth Ac-row elements
    periodic_table_78 = [
        'H',  
        'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl',  
        'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
        'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  
        'Cs', 'Ba', 
                    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
                          'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 
    ]


def composition_one_hot(data, method='percentage', index='z_number_78', only_elem_present=False):
    '''
    Make the composition to fix size vector like one hot array
    Note that this function is ONLY suitable for formula with integer number
    of each atom, i.e. a fomula with fractional occuption is not supported

    !!!REINDEX FOR ONLY_TYPE=FASLE NOT IMPLEMENT!!!

    Args:
        data: json dataset of structures from Materials Project
        Method: how the number of atoms is encoded
            percentage: (default) the number of atoms given as percentage, 
                i.e. the sum of the whole formula is 1
            formula: numbers in formula
            only_type: the value of each atom is set to 1
        index: see function element_indice for details
            z_number_78: (default) in the order of atomic number, this is default 
                because the similarity matrix is in this order
            z_number: see periodic_table in function element_indice
            pettifor: see function element_indice for details
            modified_pettifor: see element_indice
            elem_present: the vector only contain the elements presented in the dataset
        only_element_present: Used when the vector is reindexed. If true, only keep the 
            elements presented in the dataset
    Return:
        a pandas dataframe, index is mp-ids, and columns is element symbols
        elem_symbol is a list of elements present in the dataset, in Z number order
    '''
    # define the indice by call element_indice function
    element_indice()

    pero_tab_nums = []
    mp_index = []
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        # the struct.species method give a list of each site, 
        # e.g. for SrTiO3 the output is 
        # [Element Sr, Element Ti, Element O, Element O, Element O]
        pero_tab_nums.append([ x.number for x in struct.species ])
        mp_index.append(d['task_id'])
    
    # use the counter method to one-hot the element numbers
    # considering the output of species method above
    # the data_np is typically using " method == 'formula' "
    elem_vectors = pd.DataFrame([Counter(x) for x in pero_tab_nums])
    elem_vectors = elem_vectors.fillna(0).sort_index(axis=1)

    if method == 'percentage':
        # divide by total number of atoms to the sum will be 1
        # sum all columns in each row, divide row-wise
        elem_vectors = elem_vectors.div(elem_vectors.sum(axis=1), axis=0)
    elif method == 'only_type':
        # set all the non-zero values to 1
        elem_vectors[elem_vectors != 0] = 1

    # get the order of elements in the list and element symbols
    elem_numbers = elem_vectors.columns.tolist()
    # this gives all element symbols present in the dataset
    elem_symbols = [ Element.from_Z(x).name for x in elem_numbers ]
    
    elem_vectors.columns = elem_symbols
    elem_vectors.index = mp_index

    # Now the vectors in data_np is in the order of Z number but only
    # with the elements presented in the dataset
    # we may want to reindex the data_np
    # Note the index here is accutely column names in pandas, not pandas index
    if index != 'elem_present': 
        if index == 'z_number_78':
            elem_vectors = elem_vectors.reindex(columns=periodic_table_78)
        elif index == 'z_number':
            elem_vectors = elem_vectors.reindex(columns=periodic_table)
        elif index == 'pettifor':
            elem_vectors = elem_vectors.reindex(columns=pettifor)
        elif index == 'modified_pettifor':
            elem_vectors = elem_vectors.reindex(columns=modified_pettifor)
            
        if only_elem_present:
            elem_vectors = elem_vectors.dropna()
        else:
            elem_vectors = elem_vectors.fillna(0)

    return elem_vectors, elem_symbols


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
    # define the indice by call element_indice function
    element_indice()

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
            pettifor: typically for visualization purpose, 
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
    # define the indice by call element_indice function
    element_indice()

    # note that the index_col is after the use of selection
    d = pd.read_csv(input_file, sep=' ', index_col=[0,1], usecols=[0,2,4])
    # make it a two dimensional matrix
    d = d.unstack()
    # drop the multilevel when ustack
    d.columns = d.columns.droplevel()

    if order == 'pt_number':
        index = periodic_table_78
    elif order == 'pettifor':
        index = pettifor
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
    # define the indice by call element_indice function
    element_indice()
    
    all_bond_matrix = []
    num_elements = len(periodic_table_78)
    zeor_bond_matrix = pd.DataFrame(np.zeros((num_elements, num_elements)), 
                                    index=periodic_table_78, columns=periodic_table_78)
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

