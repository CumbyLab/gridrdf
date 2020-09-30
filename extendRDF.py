
import sys
import numpy as np
import pprint
import argparse
import logging
import itertools 
from pymatgen import Structure, Lattice


def extend_structure(structure, max_dist=10):
    '''
    Make supercell based on the cutoff radius(i.e. the maximum distance).  
    Get all the inequivalent sites in a primitive cell inside a supercell.  
    Typically this primtive cell locates in the center of a supercell.

    Args:
        struct: pymatgen structure
        max_dist: cutoff radius in Angstrom
    Return:
        extend_stru: extended structure, i.e. the supercell
        prim_cell_list: a list of atom indexes of the centered primtive cell
    '''
    # get primitive structure 
    # to drop the atomic species information, uncomment the following
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
    Get pair distance in the supercell.  
    One atoms must be in the selected primtive cell.  
    Currently the atomic species information is dropped.  

    Args:
        structure: pymatgen structure, typically a supercell
        max_dist: cutoff of the atomic pair distance
        prim_cell_list: index of the atoms of the selected primitive cell
    Return:
        A sortted list of atomic pair distance 
    '''
    raw_rdf = []
    for site in prim_cell_list:
        for pair_site in structure.get_neighbors(site=structure[site], r=max_dist):
            raw_rdf.append(round(pair_site[1],3))
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


def rdf_one_hot_conversion(raw_rdf, max_dist=10, npoints=100):
    '''
    convert rdf into one-hot encoder

    Args:
        raw_rdf: A sortted list of atomic pair distance 
        max_dist: cutoff of the atomic pair distance
        npoints: number of points for RDF
    Return:
        one hot coded rdf
    '''
    rdf_num = len(raw_rdf)
    rdf_index = [ int(raw_rdf[i] / max_dist * npoints) for i in range(rdf_num) ]
    rdf = np.zeros((rdf_num, npoints + 1))
    rdf[np.arange(rdf_num),rdf_index] = 1
    return rdf


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

    # converse the rdf_atom into rdf in each shell,
    # and only keep the distance values
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


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Calculate RDF with atoms')
    parse.add_argument('--input', type=str, required=False,
                        help='Input CIF containing the crystal structure')
    parse.add_argument('--output', type=str, default='rdf_bin',
                        help='Output RDF')
    parse.add_argument('--max_dist', type=float, default=10.0,
                        help='Cutoff distance of the RDF')

    args = parse.parse_args()
    infile = args.input
    outfile = args.output
    max_dist = args.max_dist

    if infile:
        # read a structure from cif to a pymatgen structure
        struct = Structure.from_file(filename=infile, primitive=True)
    else:
        # if a input structure is not provide, the code is in test mode
        # and nacl structure will be used for test propose
        nacl = Structure.from_spacegroup('Fm-3m', Lattice.cubic(5.6), 
                                        ['Na', 'Cl'], [[0.5, 0.5, 0.5], [0, 0, 0]])
        struct = nacl.get_primitive_structure()
    
    prim_cell_list = list(range(len(struct)))
    rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
                                    max_dist=max_dist)
    #rdf_bin = rdf_histo(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_size=0.1)
    rdf_bin, atom_pairs = rdf_stack_histo(rdf_atoms=rdf_atoms, structure=struct, 
                                        max_dist=max_dist, bin_size=0.1)

    np.set_printoptions(threshold=sys.maxsize) # print the whole array
    # transpose the ndarray for import into the plot program
    print(atom_pairs)
    np.savetxt(outfile, rdf_bin.transpose(), delimiter=" ",fmt='%i')


# Blow is old version of the code
# It seems that pymatgen.Site.get_neighbors method automately create the extend supercell, 
# so there is no need to extend the cell
if False:
    # make supercell and find the 'centered' primitive cell
    extend_stru, prim_cell_list = extend_structure(structure=struct, max_dist=max_dist)
    # get RDF as sorted list
    raw_rdf = get_raw_rdf(structure=extend_stru, prim_cell_list=prim_cell_list, max_dist=max_dist)
    rdf = rdf_one_hot_conversion(raw_rdf=raw_rdf, max_dist=10, npoints=100)

    with open ('raw_rdf', 'w') as f:
        pprint.pprint(raw_rdf, f)
 



