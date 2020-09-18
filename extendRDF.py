
import sys
import numpy as np
import pprint
from pymatgen import Structure


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
    # get primitive structure regardless of the atomic species
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


if __name__ == '__main__':

    max_dist = 10
    # read a structure from cif to a pymatgen structure
    struct = Structure.from_file(filename='NaCl.cif', primitive=True)
    # make supercell and find the 'centered' primitive cell
    extend_stru, prim_cell_list = extend_structure(structure=struct, max_dist=max_dist)
    # get RDF as sorted list
    raw_rdf = get_raw_rdf(structure=extend_stru, prim_cell_list=prim_cell_list, max_dist=max_dist)
    rdf = rdf_one_hot_conversion(raw_rdf=raw_rdf, max_dist=10, npoints=100)

    np.set_printoptions(threshold=sys.maxsize) # print the whole array
    with open ('raw_rdf', 'w') as f:
        pprint.pprint(raw_rdf, f)
    with open ('rdf', 'w') as f:
        pprint.pprint(rdf, f)



