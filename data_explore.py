
import numpy as np
import pandas as pd
import json
import time
import gzip
import argparse
from collections import Counter
from itertools import combinations
from pandas import DataFrame
from pymatgen import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from sklearn.preprocessing import MultiLabelBinarizer
try:
    from matminer.featurizers.structure import StructuralComplexity
except:
    print('matminer is not installed, cannot calculate structural complexity')

from extendRDF import rdf_histo, rdf_kde, get_rdf_and_atoms, shell_similarity
from otherRDFs import origin_rdf_histo


def batch_rdf(data, max_dist=10, bin_size=0.1, method='bin', normalize=True, 
                gzip_file=False):
    '''
    Read structures and output the extend RDF

    Args:
        data: input data from Materials Project
        max_dist: cut off distance of RDF
        method:
            bin: binned rdf, equal to the 'uniform' method in kde,  
                note that the 
            kde: use kernel density estimation to give a smooth curve
        gzip_file: in case RDF files take too much disk space, set 
                    this to true to gzip the files (not yet test)
    Return:

    '''
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        prim_cell_list = list(range(len(struct)))
        rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
                                        max_dist=max_dist)
        
        if method == 'kde':
            rdf_bin = rdf_kde(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_size=bin_size)
        elif method == 'bin':
            # this should be replaced by the general kde uniform method in the future
            rdf_bin = rdf_histo(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_size=bin_size)
            if normalize:
                rdf_bin = rdf_bin / len(struct)
        else:
            print('This method is not supported in RDF calculation ')

        outfile = d['task_id']
        if gzip_file:
            with gzip.open(outfile+'.gz', 'w') as f:
                # not yet test, need test before use
                f.write(rdf_bin.tostring())
        else:
            np.savetxt(outfile, rdf_bin, delimiter=' ', fmt='%.3f')
        time.sleep(0.1)
    return


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
                nbins = len(all_rdf[i][0])
                all_rdf[i] = np.append( all_rdf[i], 
                                        [[0.0] * nbins] * (trim-rdf_len), 
                                        axis=0 )
            else:
                all_rdf[i] = all_rdf[i][:trim]
    elif trim == 'none':
        pass 
    else:
        print('wrong value provided for trim') 

    return np.stack(all_rdf)


def rdf_flatten(all_rdf):
    '''
    If the rdf is not 1d, make it 1d for machine learning input

    Args:
        all_rdf: a 2D or 3D np array (rdfs) with the same length, 
            with the first dimension be number of samples
    Return:
        a 2D np array of rdfs
    '''
    if len(all_rdf[0].shape) == 2:
        all_rdf = [ x.flatten() for x in all_rdf]
    return all_rdf


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
    for j in range(len(structure)) : 
        for i in nn.get_nn_info(structure, j) :       
            bond_len.append(structure[j].distance(structure[i['site_index']]))
    bond_len = np.array(bond_len)
    return bond_len.mean(), bond_len.std()#, bond_len.ptp()


def average_coordination(structrue):
    '''
    Calculation of average coordination number over every site in a structure
    using Vorini method

    Args:
        structrue: pymatgen structure
    Return:
        Average coordination number
    '''
    ave_coord_num = []
    for atom_site in range(len(structure)) :
        ave_coord_num.append(len(nn.get_nn_info(structure, atom_site)))
    return np.array(ave_coord_num).mean()


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


def composition_one_hot(data, only_type=False, normalize=True):
    '''
    Make the composition to fix size vector like one hot array

    Args:
        data: the bulk modulus data and structure from Materials Project
        only_type: if true, only the types are included but not  
            the whole formula
        normalize: if normalize, the number of atoms in the output will 
            be given as percentage
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


def elements_selection(data, elem_list, mode='include', method='any'):
    '''
    Select a subset contains or not contain certain elements.

    Args:
        data: a list of dicts with CIFs
        elem_list: a list of the elements of interest or no-interest
        mode:
            include: select the structures have elements in the elem_list
            exclude: drop the structures have elements in the elem_list
        method:
            all: the structure has all the elements in the list
            any: the structure has any one element in the list
    Return:
    '''
    for d in data[:]:
        elements = Structure.from_str(d['cif'], fmt='cif').symbol_set
        if mode == 'include':
            if set(elem_list).isdisjoint(elements):
                data.remove(d)
        elif mode == 'exclude':
            if not set(elem_list).isdisjoint(elements):
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
    # note that the index_col is after the use selection
    d = pd.read_csv(input_file, sep=' ', index_col=[0,1], usecols=[0,2,4])
    d = d.unstack()
    # drop the multilevel when ustack
    d.columns = d.columns.droplevel()

    # Pettifor order of elements
    if order == 'pt_number':
        index = ['H', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F', 'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 
                'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
                'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
    elif order == 'pettifor':
        index = ['Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Yb', 'Eu', 'Y',  'Sc', 'Lu', 'Tm', 'Er', 'Ho', 
                'Dy', 'Tb', 'Gd', 'Sm', 'Pm', 'Nd', 'Pr', 'Ce', 'La', 'Zr', 'Hf', 'Ti', 'Nb', 'Ta', 'V',  'Mo', 
                'W',  'Cr', 'Tc', 'Re', 'Mn', 'Fe', 'Os', 'Ru', 'Co', 'Ir', 'Rh', 'Ni', 'Pt', 'Pd', 'Au', 'Ag', 
                'Cu', 'Mg', 'Hg', 'Cd', 'Zn', 'Be', 'Tl', 'In', 'Al', 'Ga', 'Pb', 'Sn', 'Ge', 'Si', 'B',  'Bi', 
                'Sb', 'As', 'P',  'Te', 'Se', 'S', 'C', 'I', 'Br', 'Cl', 'N', 'O', 'F', 'H']
    # reindex, i.e. change the order of column and rows to Pettifor order  
    d = d.fillna(0)
    d = d.reindex(columns=index, fill_value=0) 
    d = d.reindex(index=index, fill_value=0)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data explore',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', type=str, default='../MP_modulus.json',
                        help='the bulk modulus and structure from Materials Project')
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
                            '   subset: select a subset which have specified elements'
                      )
    parser.add_argument('--elem_list', type=str, default='O',
                        help='only used for subset task')
    parser.add_argument('--max_dist', type=float, default=10.0,
                        help='Cutoff distance of the RDF')

    args = parser.parse_args()
    infile = args.input_file
    rdf_dir = args.rdf_dir
    task = args.task

    elem_list = args.elem_list
    max_dist = args.max_dist

    with open(infile,'r') as f:
        data = json.load(f)

    if task == 'num_shell':
        results = num_of_shells(data=data, dir=rdf_dir)
        outfile = '../num_shell'
        np.savetxt(outfile, results, delimiter=' ', fmt='%.3f')
    elif task == 'extend_rdf_bin':
        batch_rdf(data, max_dist=max_dist, method='bin')
    elif task == 'extend_rdf_kde':
        batch_rdf(data, max_dist=max_dist, method='kde')
    elif task == 'origin_rdf':
        origin_rdf_histo(data, max_dist=max_dist)
    elif task == 'composition':
        elements_count(data)
    elif task == 'bonding_type':
        for d in data:
            struct = Structure.from_str(d['cif'], fmt='cif')
            d['bond_elem_list'], d['bond_num_list'] = bonding_type(struct)
        with open(infile.replace('.json','_with_bond.json'), 'w') as f:
            json.dump(data, f, indent=1)
    elif task == 'subset':
        print(elem_list)
        subset = elements_selection(data, elem_list=elem_list.split(), 
                                    mode='exclude', method='any')
        # note that 'data' is also changed because it is defined in __main__
        with open('subset.json', 'w') as f:
            json.dump(subset, f, indent=1)
    else:
        print('This task is not supported')

