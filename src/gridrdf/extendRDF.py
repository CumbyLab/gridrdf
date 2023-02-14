'''
Generate Radial Distribution Functions (including GRID) from a pymatgen structure.

Various functions to compute pairwise atomic distances and bin
them into histograms, with optional Gaussian broadening.


The __name__ == '__main__' part in this module is used for test runs, 
so the input_file should contain a single structure
Batch calculation of the whole dataset, i.e. multiple input structures is done in the
__name__ == '__main__' part in data_explore.py

'''

import sys
import numpy as np
import argparse
import logging
import itertools
from scipy.stats import gaussian_kde

# Handle pymatgen v2022 changes
try:
    from pymatgen import Structure
except ImportError:
    from pymatgen.core.structure import Structure   
try:
    from pymatgen import Lattice
except ImportError:
    from pymatgen.core.lattice import Lattice
from sklearn.neighbors import KernelDensity
from pyemd import emd

import warnings

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
        A sorted 1d list of atomic pair distance 
    '''
    raw_rdf = []
    for site in prim_cell_list:
        for pair_site in structure.get_neighbors(site=structure[site], r=max_dist):
            raw_rdf.append(round(pair_site[1], 3))
    return sorted(raw_rdf)

def _sorted_neighbours(structure, max_dist):
    """
    Return the sorted neighbours of each site in a Structure.

    Parameters
    ----------
    structure : PyMatGen Structure

    max_dist: float
        Maximum cutoff distance for neighbour search in Angstroms

    Returns
    -------

    neighbours : list of list
        list of pymatgen PeriodicSite objects in distance order, of format
        [[site_0_neighbours], [site_1_neigbours], ...]

    """

    # PyMatGen returns all neighbours for all sites
    neighbours = structure.get_all_neighbors(max_dist)

    for site in neighbours:
        # Sort based on neighbour.nn_distance
        site.sort(key = lambda x: x.nn_distance)

    return neighbours

def _estimate_cutoff(structure, num_neighbours):
    """
    Estimate the cutoff required to achieve a certain number of neighbours based on atom density.

    Parameters
    ----------
    structure : PyMatGen structure
    num_neighbours : int
        Number of neighbours desired

    Returns
    -------
    cutoff : float
        Minimum suggested cutoff to achieve number of neighbours

    """

    atom_density = len(structure) / structure.volume

    # Cutoff based on number of atoms expected within sphere
    min_cutoff = ((3*num_neighbours) / (4*np.pi*atom_density))**(1./3)

    return min_cutoff * 1.1

def _estimate_neighbours(structure, cutoff):
    """
    Quickly estimate the minimum number of neighbours expected within a radius based on atom density.

    Parameters
    ----------
    structure : PyMatGen structure
    cutoff : float
        Cutoff distance in Angstrom

    Returns
    -------
    min_neighbours : int
        Minimum neighbours expected for a given cutoff

    See Also
    --------
    _estimate_cutoff

    """    

    atom_density = len(structure) / structure.volume
    return int(4/3*np.pi*cutoff**3 * atom_density)

def get_pairwise_distances(structure,
                           num_neighbours=None,
                           cutoff=None,
                           return_cutoff = False):
    """
    Return a sorted list of pairwise distances for a structure up to a cutoff.

    Parameters
    ----------

    structure : PyMatGen structure

    num_neighbours : int, default None
        Number of desired neighbours

    cutoff : float, default None
        Distance cutoff (in Ang) to calculate distance to

    NOTE: Only one of num_neighbours or cutoff must be supplied

    return_cutoff: bool, default False
        Return the distance cutoff used (usually helpful when num_neighbours is supplied)

    Returns
    -------

    neighbours : list of list
        list of pymatgen PeriodicSite objects in distance order, of format
        [[site_0_neighbours], [site_1_neigbours], ...]
    cutoff : float, optional
        distance cutoff used to generate neighbours

    Notes
    -----

    - If `num_neighbours` is defined, `return_cutoff` will give the cutoff required to return the same (minimum) number
      of neighbours. Please note that for structures with large numbers of sites, this may not yield the same number 
      of neighbours for all sites (use `num_neighbours` to give that behaviour).

    """

    assert (num_neighbours is None) ^ (cutoff is None), "You must specify only one of num_neighbours or cutoff."

    if num_neighbours:
        cutoff = _estimate_cutoff(structure, num_neighbours)
    
    complete = False
    while not complete:
        neighbours = _sorted_neighbours(structure, cutoff)
        if cutoff:
            complete = True
        elif num_neighbours:
            min_count = min([len(i) for i in neighbours])
            if min_count < num_neighbours:
                cutoff *= 1.1
            else:
                complete = True

    # Ensure we are returning the correct number of neighbours
    if num_neighbours:
        for i, site in enumerate(neighbours):
            neighbours[i] = site[:num_neighbours]
        # Reset the cutoff to give the correct (minimum) number of neighbours across all sites
        cutoff = max([i[-1].nn_distance for i in neighbours]) + 0.0001
    
    if return_cutoff:
        return neighbours, cutoff
    else:
        return neighbours

def _neighbours_to_dists(neighbours):
    """
    Return the distances and occupancies from a Neighbour list.

    Parameters
    ----------

    neighbours : list of lists
        List of PyMatGen neighbour objects, ordered by site

    Returns
    -------

    dists : array-like
        Pairwise distances ordered as (sites, neighbours)
    occupancies : array-like
        Site occupancies ordered as (sites, neighbours)

    """

    simplified = []
    for site in neighbours:
        simplified.append([])
        for neig in site:
            simplified[-1].append([neig.nn_distance, neig.species.num_atoms])
    vals = np.array(simplified)

    # return distances and occupancies
    return vals[:,:,0], vals[:,:,1]

def __sample_dists_convolve(dists, weights, sampling_width, smearing):
    """ 
    Discretize distances and then convolve them with a Gaussian snearing.

    Bins the observed distances onto a fine grid, and then convolves the resulting
    histograms with a Gaussian broadening. The resulting fine grid is intended to be
    binned to a coarser distribution in order to approximate the integral

    Parameters
    ----------
    dists : array-like
        Array of pairwise distances.
    weights : array-like
        Weights for each distance (i.e. based on occupancies) with the same shape as dists
    sampling_width : float
        Width to use for fine sampling of smearing
    smearing : float
        Gaussian bandwidth to apply to distances

    Returns
    -------

    fine_bins : array-like
        Distances at which the Gaussian KDE has been sampled
    weights : array-like
        Weights for each distance (same shape as fine_bins)
    """
    
    from scipy.ndimage import gaussian_filter

    max_dist = dists.max()
    # Set up an array of finely spaced distances
    fine_bins = np.array([np.arange(0, max_dist+3*smearing+sampling_width, sampling_width)]*dists.shape[0])
    fine_counts = np.zeros((fine_bins.shape[0], fine_bins.shape[1]-1))

    for shell in np.arange(dists.shape[0]):
        fine_counts[shell,:] = np.histogram(dists[shell],
                                             bins=fine_bins[shell],
                                             weights=weights[shell],
                                             )[0]    

    # Calculate the filtered version. Note that scipy works in "pixel" units,
    # so the standard deviation is speified in terms of number of bins.
    # 
    # Should gaussian_filter take the std rather than std^2 of the Gaussian?
    smeared_wts = gaussian_filter(fine_counts, (0, smearing/sampling_width))

    return fine_bins[:, :-1], smeared_wts

def __sample_kde_fine(dists, weights, sampling_width, smearing):
    """ 
    Compute KDE estimate of broadening for each distance, and bin based on sampling_width.

    Parameters
    ----------
    dists : array-like
        Array of pairwise distances.
    weights : array-like
        Weights for each distance (i.e. based on occupancies) with the same shape as dists
    sampling_width : float
        Width to use for fine sampling of smearing
    smearing : float
        Gaussian bandwidth to apply to distances

    Returns
    -------

    fine_bins : array-like
        Distances at which the Gaussian KDE has been sampled
    weights : array-like
        Weights for each distance (same shape as fine_bins)

    Notes
    -----

    - We use scikit-learn's `KernelDensity` rather than scipy's `gaussian_kde` because the latter does not allow
      the bandwidth to be explicitly defined (instead, it is relative to the std of the data)
    
    """

    max_dist = dists.max()
    fine_bins = np.array([np.arange(0, max_dist+3*smearing, sampling_width)]*dists.shape[0])

    fine_counts = np.zeros_like(fine_bins)

    for shell in np.arange(dists.shape[0]):
        #kde = gaussian_kde(dists[shell], bw_method=smearing)
        #fine_counts[shell,:] = kde.evaluate(fine_bins[shell,:])
        kde = KernelDensity(bandwidth=smearing, kernel='gaussian')
        kde.fit(dists[shell].reshape([-1,1]), sample_weight=weights[shell])

        fine_counts[shell] = np.exp(kde.score_samples(fine_bins[shell,:].reshape(-1,1)))

    return fine_bins, fine_counts

def calculate_rdf(structure,
                  neighbours,
                  rdf_type='grid',
                  max_dist=None,
                  bin_width=0.1,
                  smearing = 0.1,
                  normed = True,
                  broadening_method = 'convolve',
                  ):
    """
    Return the GRID (2D) RDF from a list of neighbours

    Parameters
    ----------

    structure : PyMatGen Structure
    neighbours : list
        List of neighbours generated using `get_pairwise_distances`
    rdf_type : str 
        Format of RDF to return
            'grid': return 2D RDF with distances sorted by neighbour order
            'simple': return the traditional 1D RDF
    max_dist : Float, default None
        Maximum cutoff to use when generating RDF representation. If None,
        the maximum observed distance will be used.
    bin_width : Float, default 0.1
        Width of histogram bins in Angstroms
    smearing : Float, default 0.1
        Width of Gaussian smearing to apply to distances prior to binning. If 0,
        distances will be binned without smearing
    normed : Bool, default True
        Whether to normalise the resulting histogram area(s). For 2D (grid) 
        histogram, each shell has area = 1
    broadening_method : str, default 'convolve'
        How to compute the Gaussian broadening (only applies if smearing > 0.)
        `convolve` : bin the distances into a finer distribution, and then apply a Gaussian
                     convolution to the results
        `kde`      : construct a KernelDensityEstimate based on the distances, and then 
                     compute the magnitude of this KDE on a fine distribution before binning the
                     results. 

    Notes
    -----

    **Broadening**

    Gaussian broadening is not implemented exactly due to the complexity of integrating a KDE
    across multiple bin ranges. Instead, two approaches are available which yield very similar overall
    results.
    broadening_method=`convolve` first bins the distances into a finely divided histogram (currently bin_width/5)
    before applying a Gaussian convolution across these bins. Formally this loses precision in the bond
    distances before broadening, but is otherwise correct (and fast).

    broadening_method=`kde` tries to compute a numerically correct KDE based on the exact distances, before then
    computing the value of this KDE on a fine scale (bin_width/5). The approximation is that the integral of 
    the KDE is assumed to be equal to the sum of these closely spaced KDE values. Although not strictly accurate, 
    this approximation is likely to affect every distance equally, therefore the overall histogram is probably 
    only different by a scale factor.

    The original `rdf_kde` method uses the same approach as broadening_method=`convolve`, with the exception that 
    the "fine" histogram was the same as the final histogram defined by `bin_width`. Please note that rounding was 
    also used, which may lead to slightly different numerical results.

    """

    dists, weights = _neighbours_to_dists(neighbours)
    # Arrange distances and weights into GRID-shell order
    dists = dists.T
    weights = weights.T

    # Multiply weights by central site occupancy
    weights = weights * [i.species.num_atoms for i in structure]

    # Set up vectors for distance and shell_number
    if max_dist:
        distance = np.arange(0,max_dist+bin_width, bin_width)
    else:
        max_dist = dists.max()
        distance = np.arange(0,max_dist+bin_width, bin_width)        

    shell_number = np.arange(dists.shape[0])

    if smearing > 0:
        # Calculate the gaussian broadening on a fine grid so that it can be binned later.
        # fine grid of bin_width/5 is somewhat arbitrary.
        convolve_scale = 5
        if broadening_method == 'kde':
            dists, weights = __sample_kde_fine(dists, weights, bin_width/convolve_scale, smearing)
        elif broadening_method == 'convolve':
            dists, weights = __sample_dists_convolve(dists, weights, bin_width/convolve_scale, smearing)

    if rdf_type == 'grid':
        binned_dists = np.zeros((shell_number.shape[0], distance.shape[0]-1))
        for shell in np.arange(dists.shape[0]):
            binned_dists[shell,:] = np.histogram(dists[shell],
                                                    bins=distance,
                                                    weights=weights[shell],
                                                    density=normed
                                                    )[0]

    elif rdf_type == 'simple':
        binned_dists = np.histogram(dists,
                                    bins=distance,
                                    weights = weights,
                                    density=normed
                                    )[0]
    else:
        raise ValueError(f"Unknown RDF type {rdf_type}")

    return binned_dists




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

    warnings.warn('`get_rdf_and_atoms` will be deprecated in a future version. New code should use `get_pairwise_distances` instead', FutureWarning)

    rdf_atoms = {}
    for i, site in enumerate(prim_cell_list):
        rdf_atoms[i] = []
        site1 = structure[site].species_string
        for pair_site in structure.get_neighbors(site=structure[site], r=max_dist):
            site2 = pair_site[0].species_string
            rdf_atoms[i].append([round(pair_site[1],3), site1, site2])
        rdf_atoms[i].sort()
    return rdf_atoms


def rdf_histo(rdf_atoms, max_dist=10, bin_width=0.1):
    '''
    Convert the raw rdf with atoms to binned frequencies i.e. histogram.

    Args:
        rdf_atoms: pair distance of rdf with atomic speicies (output of get_rdf_and_atoms)
        max_dist: cutoff of the atomic pair distance
        bin_width: bin size for generating counts
    Return:
        Binned rdf frequencies for each shell of neasest neighbor
    '''

    warnings.warn('`rdf_histo` will be deprecated in a future version. New code should use `calculate_rdf` instead.', FutureWarning)

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

    bins = np.linspace(start=0, stop=max_dist, num=int(max_dist/bin_width)+1)
    # np.histogram also return the bin edge, which is not needed
    # so only the bin counts [0] is kept    
    rdf_bin = [ np.histogram(x, bins=bins, density=False)[0]
                for x in rdf_nn_shells ]
    return np.array(rdf_bin)

def find_all_neighbours(structure_list,
                        num_neighbours=None,
                        cutoff=None,
                        return_limits = False,
                        dryrun = False):
    """
    Return neighbour lists for an iterable list of structures.

    Parameters
    ----------
    
    structures : list of PyMatGen structures

    num_neighbours : int, default None
        Number of desired neighbours for each structure

    cutoff : float, default None
        Distance cutoff (in Ang) to find neighbours

    NOTE: Only one of num_neighbours or cutoff must be supplied

    return_limits : bool, default False
        If True, return either [min_neigh, max_neigh] or [min_cutoff, max_cutoff] depending on which parameter
        was supplied.

    dryrun : bool, default False
        If True, only estimate cutoffs for the given parameters, but don't calculate neighbours

    Returns
    -------

    neighbours : nested lists of neighbours
        list of pymatgen PeriodicSite objects in distance order, of format
        [[site_0_neighbours], [site_1_neigbours], ...]

    """

    assert (num_neighbours is None) ^ (cutoff is None), "You must specify only one of num_neighbours or cutoff."

    # Try to quickly estimate cutoffs or neighbours
    if num_neighbours:
        est_cutoffs = [_estimate_cutoff(i, num_neighbours) for i in structure_list]
        min_cut = min(est_cutoffs)
        max_cut = max(est_cutoffs)
        warnings.warn(f'The estimated cutoff distances range from {min_cut:.2f} to {max_cut:.2f} Angstroms', stacklevel=2)
    elif cutoff:
        est_neig = [_estimate_neighbours(i, cutoff) for i in structure_list]
        min_neig = min(est_neig)
        max_neig = max(est_neig)
        warnings.warn(f'The estimated number of neighbours ranges from {min_neig} to {max_neig}.', stacklevel=2)

    if dryrun:
        return None

    neighbours = []
    neigh_range = [1E4,0]
    cutoff_range = [1E7,0]

    for struc in structure_list:
        neighbours.append(get_pairwise_distances(struc, num_neighbours=num_neighbours, cutoff=cutoff))
        if cutoff:
            current_neighbour_count = max([len(i) for i in neighbours[-1]])
            if current_neighbour_count < neigh_range[0]:
                neigh_range[0] = current_neighbour_count
            elif current_neighbour_count > neigh_range[1]:
                neigh_range[1] = current_neighbour_count
        elif num_neighbours:
            max_dist = neighbours[-1][-1][-1].nn_distance
            if max_dist < cutoff_range[0]:
                cutoff_range[0] = max_dist
            elif max_dist > cutoff_range[1]:
                cutoff_range[1] = max_dist

    #warnings.warn(f'Cutoff ranges from {cutoff_range[0]:.4f} to {cutoff_range[1]:.4f}.', stacklevel=2)
    #warnings.warn(f'Number of neighbours ranges from {neigh_range[0]} to {neigh_range[1]}.', stacklevel=2)
    
    if return_limits:
        if num_neighbours:
            return neighbours, cutoff_range
        elif cutoff:
            return neighbours, neigh_range
    else:
        return neighbours

def rdf_stack_histo(rdf_atoms, structure, max_dist=10, bin_width=0.1, bond_direct=False):
    '''
    Convert the raw rdf with atoms to binned frequencies i.e. histogram
    and condsidering different atomic pairs

    Args:
        rdf_atoms: pair distance of rdf with atomic speicies (output of `get_rdf_and_atoms`)
        structure: pymatgen structure
        max_dist: cutoff of the atomic pair distance
        bin_width: bin size for generating counts
        bond_direct: if True, same atom pairs (e.g ['Si','O'] and ['O','Si']) are merged
    Return:
        Binned rdf frequencies for each shell of nearest neighbor
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

    bins = np.linspace(start=0, stop=max_dist, num=int(max_dist/bin_width)+1)
    # np.histogram also return the bin edge, which is not needed
    # so only the bin counts [0] is kept    
    rdf_bin = [ np.histogram(x, bins=bins, density=False)[0]
                for x in rdf_atom_pair_shells ]
    return np.array(rdf_bin), atom_pair_list


def rdf_kde(rdf_atoms, max_dist=10, bin_width=0.1, bandwidth=0.1):
    '''
    Convert the raw rdf with atoms to binned frequencies with Gaussian smearing.

    Args:
        rdf_atoms: pair distance of rdf with atomic speicies (output of get_rdf_and_atoms)
        max_dist: cutoff of the atomic pair distance
        bin_width: bin size for generating counts
    Return:
        rdf_bin: Gaussian smeared rdf frequencies for each shell of neasest neighbor
    '''

    warnings.warn('`rdf_kde` will be deprecated in a future version. New code should use `calculate_rdf` instead.', FutureWarning)

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

    # the kernel density method need a 2d input, so add a new axis
    bins = np.linspace(start=0, stop=max_dist, num=int(max_dist/bin_width)+1)[:, np.newaxis]

    rdf_bin = []
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    for x in rdf_nn_shells:
        log_dens = kde.fit(np.array(x)[:, np.newaxis]).score_samples(bins)
        # Due to Gaussian smearing hitting the edge of the RDF range,
        # occasionally the densities do not sum to 1.
        # We normalize here, so Wasserstein_distance doesn't need to
        dens = np.exp(log_dens)
        dens = dens / dens.sum()
        rdf_bin.append(dens)

    return np.array(rdf_bin)


# def shell_similarity(rdf_bin):
#     '''
#     Calculate the earth mover distance (EMD) between adjacent rdf shells  
#     i.e. the first value is the EMD between the first shell and second

#     Args:
#         rdf_bin: calculated rdf, only rdf from function rdf_histo has been tested
#     Return:
#         np array of the similarity, with length (len(rdf_bin)-1)
#     '''
#     shell_dist = np.zeros((len(rdf_bin), len(rdf_bin)))
#     dist_matrix = np.ones((len(rdf_bin[0]), len(rdf_bin[0])))
#     np.fill_diagonal(dist_matrix, 0)
#     for i, r1 in enumerate(rdf_bin):
#         for j, r2 in enumerate(rdf_bin):
#             if i < j:
#                 dissim = emd(r1.astype('float64'), r2.astype('float64'), 
#                         dist_matrix.astype('float64'))
#                 shell_dist[i,j] = dissim
#                 shell_dist[j,i] = dissim
#    return shell_dist


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
        rdf_bin = rdf_histo(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_width=0.1)
        if outfile:
            np.savetxt(outfile, rdf_bin.transpose(), delimiter=' ',fmt='%i')
    
    elif task == 'stack_rdf':
        rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
                                        max_dist=max_dist)
        rdf_bin, atom_pairs = rdf_stack_histo(rdf_atoms=rdf_atoms, structure=struct, 
                                            max_dist=max_dist, bin_width=0.1)
        if outfile:
            print(atom_pairs)
            # transpose the ndarray for import into the plot program
            np.savetxt(outfile, rdf_bin.transpose(), delimiter=' ',fmt='%i')
    
    # elif task == 'shell_similarity':
    #     rdf_atoms = get_rdf_and_atoms(structure=struct, prim_cell_list=prim_cell_list, 
    #                                     max_dist=max_dist)
    #     rdf_bin = rdf_histo(rdf_atoms=rdf_atoms, max_dist=max_dist, bin_width=0.1)
    #     if trim != 0:
    #         rdf_bin = rdf_bin[:trim]

    #     shell_simi = shell_similarity(rdf_bin)
    #     print(shell_simi)
    #     if outfile:
    #         np.savetxt(outfile, shell_simi, delimiter=' ', fmt='%.3f')
    
    elif task == 'raw_rdf':
        raw_rdf = get_raw_rdf(structure=struct, prim_cell_list=prim_cell_list, max_dist=max_dist)
    else:
        print('This task is not supported')

