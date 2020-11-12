
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
from sklearn.preprocessing import MultiLabelBinarizer
try:
    from matminer.featurizers.structure import StructuralComplexity
except:
    print('matminer is not installed, cannot calculate structural complexity')

from extendRDF import rdf_histo, rdf_kde, get_rdf_and_atoms
from otherRDFs import origin_rdf_histo


def batch_rdf(data, max_dist=10, bin_size=0.1, method='bin', gzip_file=False):
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
            # due to the method used, all rdf should be normalized
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
    sc = StructuralComplexity()
    results = []
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        num_shell = sum(1 for line in open(dir + '/' + d['task_id']))
        #complex_atom, complex_cell = sc.featurize(struct)
        results.append(np.array([
            #d['elasticity.K_Voigt'],    # bulk modulus
            num_shell,                  # number of RDF shells
            struct.volume/len(struct),  # volume per atom
            struct.density,             # density in g/cm^3
            len(struct),                # number of atoms
            len(struct.symbol_set),     # number of types of elements
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
        if set(elem_list).isdisjoint(elements):
            data.remove(d)

    return data


def similarity_matrix():
    '''
    Read the original file from the below reference, and convert into a dictionary

    Hautier, G., et al. (2011). 
    "Data mined ionic substitutions for the discovery of new compounds." 
    Inorganic Chemistry 50(2): 656-663.

    Preprocess the supporting txt file in the above paper using the following

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

    Args:

    Return:

    '''
    # note that the index_col is after the use selection
    d = pd.read_csv('dist_matrix', sep=' ', index_col=[0,1], usecols=[0,2,4])
    d = d.unstack()
    # drop the multilevel when ustack
    d.columns = d.columns.droplevel()

    # Pettifor order of elements
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
    np.fill_diagonal(d.values,20)
    d = d / 20
    # then fill the zeros
    d.loc[:,'Pm'] = d.loc[:,'Sm'] # column, same as:  d['Pm'] = d['Sm']
    d.loc['Pm',:] = d.loc['Sm',:] # row
    # other zero mean very dissimilar, so set to a very small value
    d.replace(0, 0.01, inplace=True)


    

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Data explore',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parse.add_argument('--input', type=str, default='../MP_modulus.json',
                        help='the bulk modulus and structure from Materials Project')
    parse.add_argument('--rdf_dir', type=str, default='./',
                        help='dir has all the rdf files')
    parse.add_argument('--task', type=str, default='num_shell',
                        help='which property to be calculated: \n' +
                            '   num_shell: number of RDF shells and other propreties \n' +
                            '   extend_rdf_bin: binned extend RDF of all the CIFs \n' +
                            '   extend_rdf_kde: kernel density estimation RDF \n' + 
                            '   origin_rdf: calcualte vanilla RDF of all the CIFs \n' + 
                            '   composition: element-wise statistics of all compositions \n' +
                            '   subset: select a subset which have specified elements'
                      )
    parse.add_argument('--elem_list', type=float, default='O',
                        help='only used for subset task')
    parse.add_argument('--max_dist', type=float, default=10.0,
                        help='Cutoff distance of the RDF')

    args = parse.parse_args()
    infile = args.input
    rdf_dir = args.rdf_dir
    task = args.task

    elem_list = args.elem_list
    max_dist = args.max_dist

    with open(infile,'r') as f:
        data = json.load(f)

    if task == 'num_shell':
        results = num_of_shells(data=data, dir=rdf_dir)
        outfile = '../num_shell'
        np.savetxt(outfile, results, delimiter=' ',fmt='%.3f')
    elif task == 'extend_rdf_bin':
        batch_rdf(data, max_dist=max_dist, method='bin')
    elif task == 'extend_rdf_kde':
        batch_rdf(data, max_dist=max_dist, method='kde')
    elif task == 'origin_rdf':
        origin_rdf_histo(data, max_dist=max_dist)
    elif task == 'composition':
        elements_count(data)
    elif task == 'subset':
        print(elem_list)
        subset = elements_selection(data, elem_list=elem_list.split(), 
                                    mode='include', method='any')
        # note that 'data' is also changed because it is defined in __main__
        with open('subset.json', 'w') as f:
            json.dump(subset, f, indent=1)
    else:
        print('This task is not supported')

