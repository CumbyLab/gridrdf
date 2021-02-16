'''
The __name__ == '__main__' part in this module is used for test runs, 
so the input_file should contain a single structure
Batch calculation of the whole dataset, i.e. multiple input structures is done in the
__name__ == '__main__' part in data_explore.py

NB1: The 'prim_cell_list' variable is used with the 'extend_structure' function, 
    when the function is deprecated, this variable is kept maybe useful in the future
'''

import sys
import numpy as np
import argparse
import logging
import itertools 
from pymatgen import Structure, Lattice
from sklearn.neighbors import KernelDensity
from pyemd import emd


def extend_structure(structure, max_dist=10):
    '''
    Currently DEPRECATED! because that
    pymatgen.Site.get_neighbors method automately create the extend supercell, 
    so there is no need to extend the cell

    Make supercell based on the cutoff radius(i.e. the maximum distance).  
    Get all the inequivalent sites in a primitive cell inside a supercell.  
    Typically this primtive cell locates in the center of a supercell.

    Usage: make supercell and find the 'centered' primitive cell
    extend_stru, prim_cell_list = extend_structure(structure=struct, max_dist=max_dist)

    Args:
        struct: pymatgen structure
        max_dist: cutoff radius in Angstrom
    Return:
        extend_stru: extended structure, i.e. the supercell
        prim_cell_list: a list of atom indexes of the centered primtive cell
    '''
    # get primitive structure 
    # to drop the atomic species information, uncomment the following line
    #struct[:] = 'X'
    prim_stru = struct.get_primitive_structure()

    # extend the structure
    latt = prim_stru.lattice.matrix
    latt_cross = np.array([ np.cross(latt[1], latt[2]), 
                            np.cross(latt[2], latt[0]), 
                            np.cross(latt[0], latt[1]) ])   
    latt_proj = np.array([ np.dot(latt[i], latt_cross[i]) 
                            / np.dot(latt_cross[i], latt_cross[i]) 
                            * latt_cross[i] for i in range(3) ])
    latt_proj_norm = np.array([ np.linalg.norm(latt_proj[i]) for i in range(3) ])
    scale_factor = [int(np.ceil(max_dist / latt_proj_norm[i]) + 1) * 2 for i in range(3)]

    # get frac coordinates of the primitive cell in the supercell
    prim_sites = []
    for prim_site in prim_stru.frac_coords:
        prim_sites.append(prim_site / scale_factor + 0.5)

    prim_stru.make_supercell(to_unit_cell=False, 
                            scaling_matrix=[[scale_factor[0], 0, 0],
                                            [0, scale_factor[1], 0],
                                            [0, 0, scale_factor[2]]])
    # the make_supercell method only modifies the given structure 
    # rather than create a new structure
    extend_stru = prim_stru

    # find which atoms are in the selected primtive cell
    prim_cell_list = []
    for i, extend_site in enumerate(extend_stru.frac_coords):
        for prim_site in prim_sites:
            if np.allclose(extend_site, prim_site, rtol=1e-3):
                prim_cell_list.append(i)

    return extend_stru, prim_cell_list


def get_raw_rdf(structure, prim_cell_list, max_dist=10):
    '''
    Get pair distance in the structure at a given cutoff.
    This is the raw pair distance values before binning.    
    Currently the atomic species information is dropped.  

    Args:
        structure: pymatgen structure, typically a supercell
        max_dist: cutoff of the atomic pair distance
        prim_cell_list: index of the atoms of the selected primitive cell 
            (See NB1 in the header of this file)
    Return:
        A sortted 1d list of atomic pair distance 
    '''
    raw_rdf = []
    for site in prim_cell_list:
        for pair_site in structure.get_neighbors(site=structure[site], r=max_dist):
            raw_rdf.append(round(pair_site[1], 3))
    return sorted(raw_rdf)


def get_rdf_and_atoms(structure, prim_cell_list, max_dist=10):
    '''
    Get pair distance in the supercell, and the element symbols of the atom pair.  
    One atoms must be in the selected primtive cell.  
    The output dictionary should be like this:  
    {0: [[1.564, 'Si', 'O'],  # where '0' is the atom number
        [1.592, 'Si', 'O'],
        [1.735, 'Si', 'O'],
        [1.775, 'Si', 'O'],
        [2.924, 'Si', 'Si'],
        [3.128, 'Si', 'Si'],
        [3.148, 'Si', 'Si'], ...... } # list all pair with atom 0 within cutoff

    Args:
        structure: pymatgen structure, typically a supercell
        max_dist: cutoff of the atomic pair distance
        prim_cell_list: index of the atoms of the selected primitive cell
            (See NB1 in the header of this file)
    Return:
        A sortted list of atomic pair distance, with atom species
    '''
    rdf_atoms = {}
    for i, site in enumerate(prim_cell_list):
        rdf_atoms[i] = []
        site1 = structure[site].species_string
        for pair_site in structure.get_neighbors(site=structure[site], r=max_dist):
            site2 = pair_site[0].species_string
            rdf_atoms[i].append([round(pair_site[1],3), site1, site2])
        rdf_atoms[i].sort()
    return rdf_atoms


def rdf_histo(rdf_atoms, max_dist=10, bin_size=0.1):
    '''
    Convert the raw rdf with atoms to binned frequencies i.e. histogram

    Args:
        rdf_atoms: pair distance of rdf with atomic speicies (output of get_rdf_and_atoms)
        max_dist: cutoff of the atomic pair distance
        bin_size: bin size for generating counts
    Return:
        Binned rdf frequencies for each shell of neasest neighbor
    '''
    # get the longest rdf number
    rdf_count = [ len(x) for x in rdf_atoms.values() ]
    rdf_len = np.array(rdf_count).max()

    # converse the rdf_atom into rdf in each shell, and only keep the distance values
    # e.g. rdf_nn_shell[0] contain all the pair distance of the first NN
    rdf_nn_shells = []
    for x in range(rdf_len):
         rdf_nn_shells.append( [line[x][0] 
                            for line in rdf_atoms.values() 
                            if len(line) > x] )

    bins = np.linspace(start=0, stop=max_dist, num=int(max_dist/bin_size)+1)
    # np.histogram also return the bin edge, which is not needed
    # so only the bin counts [0] is kept    
    rdf_bin = [ np.histogram(x, bins=bins, density=False)[0]
                for x in rdf_nn_shells ]
    return np.array(rdf_bin)


def rdf_stack_histo(rdf_atoms, structure, max_dist=10, bin_size=0.1, bond_direct=False):
    '''
    Convert the raw rdf with atoms to binned frequencies i.e. histogram
    and condsidering different atomic pairs

    Args:
        rdf_atoms: pair distance of rdf with atomic speicies (output of get_rdf_and_atoms)
        structure: pymatgen structure
        max_dist: cutoff of the atomic pair distance
        bin_size: bin size for generating counts
        bond_direct: if True, same atom pairs (e.g ['Si','O'] and ['O','Si']) are merged
    Return:
        Binned rdf frequencies for each shell of neasest neighbor
        and a string of ordered atomic pairs
    '''
    # get the longest rdf number
    rdf_count = [ len(x) for x in rdf_atoms.values() ]
    rdf_len = np.array(rdf_count).max()

    # converse the rdf_atom into rdf in each shell,
    # i.e. rdf_nn_shell[0] contain all the pair distance of the first NN
    rdf_nn_shells = []
    for x in range(rdf_len):
         rdf_nn_shells.append([ line[x] 
                            for line in rdf_atoms.values() 
                            if len(line) > x ])

    # breakdown each rdf_shell to atom pair dependent
    rdf_atom_pair_shells = []
    if bond_direct:
        # get all the atomic pairs
        atom_pair_list = list(itertools.product(structure.symbol_set, repeat=2))
        for rdf_shell in rdf_nn_shells:
            for atom_pair in atom_pair_list:
                rdf_atom_pair_shells.append([ x[0]
                                            for x in rdf_shell
                                            if x[1:] == list(atom_pair) ])
    else:
        atom_pair_list = list(itertools.combinations(structure.symbol_set, r=2)) \
                        + [ (a,a) for a in structure.symbol_set ]
        for rdf_shell in rdf_nn_shells:
            for atom_pair in atom_pair_list:
                rdf_atom_pair_shells.append([ x[0]
                                            for x in rdf_shell
                                            if (x[1:] == list(atom_pair) or 
                                                x[1:][::-1] == list(atom_pair)) ])    

    bins = np.linspace(start=0, stop=max_dist, num=int(max_dist/bin_size)+1)
    # np.histogram also return the bin edge, which is not needed
    # so only the bin counts [0] is kept    
    rdf_bin = [ np.histogram(x, bins=bins, density=False)[0]
                for x in rdf_atom_pair_shells ]
    return np.array(rdf_bin), atom_pair_list


def rdf_kde(rdf_atoms, max_dist=10, bin_size=0.1, bandwidth=0.1):
    '''
    Convert the raw rdf with atoms to binned frequencies with Gaussian smearing

    Args:
        rdf_atoms: pair distance of rdf with atomic speicies (output of get_rdf_and_atoms)
        max_dist: cutoff of the atomic pair distance
        bin_size: bin size for generating counts
    Return:
        Gaussian smeared rdf frequencies for each shell of neasest neighbor
    '''
    # get the longest rdf number
    rdf_count = [ len(x) for x in rdf_atoms.values() ]
    rdf_len = np.array(rdf_count).max()

    # converse the rdf_atom into rdf in each shell,
    # and only keep the distance values
    # e.g. rdf_nn_shell[0] contain all the pair distance of the first NN
    rdf_nn_shells = []
    for x in range(rdf_len):
         rdf_nn_shells.append( [line[x]
                            for line in rdf_atoms.values() 
                            if len(line) > x] )

    # the kernel density method need a 2d input, so add a new axis
    bins = np.linspace(start=0, stop=max_dist, num=int(max_dist/bin_size)+1)[:, np.newaxis]

    rdf_bin = []
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    for x in rdf_nn_shells:
        log_dens = kde.fit(np.array(x)[:, np.newaxis]).score_samples(bins)
        rdf_bin.append(np.exp(log_dens))

    return np.array(rdf_bin)


def shell_similarity(rdf_bin):
    '''
    Calculate the earth mover distance (EMD) between adjacent rdf shells  
    i.e. the first value is the EMD between the first shell and second

    Args:
        rdf_bin: calculated rdf, only rdf from function rdf_histo has been tested
    Return:
        np array of the similarity, with length (len(rdf_bin)-1)
    '''
    shell_dist = np.zeros((len(rdf_bin), len(rdf_bin)))
    dist_matrix = np.ones((len(rdf_bin[0]), len(rdf_bin[0])))
    np.fill_diagonal(dist_matrix, 0)
    for i, r1 in enumerate(rdf_bin):
        for j, r2 in enumerate(rdf_bin):
            if i < j:
                dissim = emd(r1.astype('float64'), r2.astype('float64'), 
                        dist_matrix.astype('float64'))
                shell_dist[i,j] = dissim
                shell_dist[j,i] = dissim
    return shell_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate RDF with atoms',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input CIF containing the crystal structure')
    parser.add_argument('--task', type=str, default='shell_similarity',
                        help='what to be calculated: \n' +
                            '   rdf: calculate RDF \n' +
                            '   stack_rdf: RDF with different atomic pair \n' +
                            '   shell_similarity: the similarity between two nearest shell \n' +
                            '   raw_rdf: origin 1D RDF as sorted list'
                      )
    parser.add_argument('--output', type=str, default=None,
                        help='Output RDF')
    parser.add_argument('--max_dist', type=float, default=10.0,
                        help='Cutoff distance of the RDF')
    parser.add_argument('--trim', type=int, default=30,
                        help='the number of shells for RDF, 0 means no trim')

    args = parser.parse_args()
    input_file = args.input_file
    task = args.task
    outfile = args.output
    max_dist = args.max_dist
    trim = args.trim

    np.set_printoptions(threshold=sys.maxsize) # print the whole array

    if input_file:
        # read a structure from cif to a pymatgen structure
        struct = Structure.from_file(filename=input_file, primitive=True)
    else:
        # if a input structure is not provide, the code is in test mode
        # and nacl structure will be used for test propose
        nacl = Structure.from_spacegroup('Fm-3m', Lattice.cubic(5.6), 
                                        ['Na', 'Cl'], [[0.5, 0.5, 0.5], [0, 0, 0]])
        struct = nacl.get_primitive_structure()
    
    # The 'prim_cell_list' is used with the 'extend_structure' function, when the function
    # is deprecated, this variable is kept maybe useful in the future
    prim_cell_list = list(range(len(struct)))

    if task == 'rdf':
        rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
                                        max_dist=max_dist)
        rdf_bin = rdf_histo(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_size=0.1)
        if outfile:
            np.savetxt(outfile, rdf_bin.transpose(), delimiter=' ',fmt='%i')
    
    elif task == 'stack_rdf':
        rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
                                        max_dist=max_dist)
        rdf_bin, atom_pairs = rdf_stack_histo(rdf_atoms=rdf_atoms, structure=struct, 
                                            max_dist=max_dist, bin_size=0.1)
        if outfile:
            print(atom_pairs)
            # transpose the ndarray for import into the plot program
            np.savetxt(outfile, rdf_bin.transpose(), delimiter=' ',fmt='%i')
    
    elif task == 'shell_similarity':
        rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
                                        max_dist=max_dist)
        rdf_bin = rdf_histo(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_size=0.1)
        if trim != 0:
            rdf_bin = rdf_bin[:trim]

        shell_simi = shell_similarity(rdf_bin)
        print(shell_simi)
        if outfile:
            np.savetxt(outfile, shell_simi, delimiter=' ', fmt='%.3f')
    
    elif task == 'raw_rdf':
        raw_rdf = get_raw_rdf(structure=extend_stru, prim_cell_list=prim_cell_list, max_dist=max_dist)
    else:
        print('This task is not supported')

