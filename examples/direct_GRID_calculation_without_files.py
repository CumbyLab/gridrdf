"""
An example of how to compute GRID distances directly without writing intermediate files.

This example uses manually created PyMatGen Structure objects, but could be modified to 
compute Structures from another source.

NOTE: If the number of Structures gets too large, this could have memory implications.

"""

__author__ = "James Cumby"
__email__ = "james.cumby@ed.ac.uk"

import gridrdf
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
import os
import tqdm
import numpy as np
import pandas as pd

script_loc = os.path.dirname(__file__)
data_source_loc = os.path.join(script_loc, '../data_sources')

# Variables to define cutoffs etc
maximum_grid_distance = 10
bin_size = 0.1
broadening = 0.1
number_of_shells = 100

# Set up dummy pymatgen Structures with different cells/compositions
dummy_structures = [
    Structure(Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60), ["Si", "Si"], [[0, 0, 0], [0.75, 0.5, 0.75]]),
    Structure(Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60), ["Ni", "Ni"], [[0, 0, 0], [0.75, 0.5, 0.75]]),
    Structure(Lattice.from_parameters(a=3.94, b=3.94, c=3.94, alpha=120, beta=90, gamma=60), ["Si", "Si"], [[0, 0, 0], [0.75, 0.5, 0.75]]),
]


# First (and slowest), find the ordered distances to neighbouring atoms.
# Note that we specify a certain number of neighbours (number_of_shells) to simplify processing,
# but a distance cutoff could be used instead.
neighbours, cutoffs = gridrdf.extendRDF.find_all_neighbours(dummy_structures,
                                                            num_neighbours=number_of_shells,
                                                            cutoff=None,
                                                            return_limits = True,
                                                            dryrun = False)
    
# Adjust the cutoff so that all nearest neighbours can fit
max_dist = max(maximum_grid_distance, np.round(cutoffs[1], 1))
if max_dist != maximum_grid_distance:
    print(f"Maximum distance has been updated to {max_dist} to account for {number_of_shells} neighbours")

# Empty list to hold GRID arrays
grid_representations = []

# Iteratively calculate GRIDs

# For memory-intensive processing (i.e. large numbers of structures)
# `return_sparse` can be used to reduce the size of each GRID representation (as a sparse array)
for i, struct in tqdm.tqdm(enumerate(dummy_structures)):
    grid_representations.append(gridrdf.extendRDF.calculate_rdf(struct,
                                                                neighbours[i],
                                                                rdf_type='grid',
                                                                max_dist=max_dist,
                                                                bin_width=bin_size,
                                                                smearing = broadening,
                                                                normed = True,
                                                                broadening_method = 'convolve',
                                                                return_sparse = False)
                                )


    
# Calculate EMD similarity between grids using a numba-optimised EMD calculation

# If the 2D dissimilarity matrix is large, `results_array` can be used to store results efficiently,
# for instance by passing a h5py dataset object in order to write straight to disk.

# Note: This is very slow for this example due to numba compilation, but is 
# optimised for larger numbers of structures
grid_similarity = gridrdf.earth_mover_distance.super_fast_EMD_matrix(grid_representations,
                                                                     bin_width = bin_size,
                                                                    )    

grid_similarity = pd.DataFrame(grid_similarity)



# Currently, composition_one_hot requires a list of dicts, each with a 'task_id' for each structure
# NOTE - this is likely to change in a future release
structure_ids = [ {'task_id': i, 'structure':dummy_structures[i]} for i in range(len(grid_representations)) ]

# First, convert composition to vector encoding
elem_vectors, elem_symbols = gridrdf.composition.composition_one_hot(structure_ids, method='percentage')

# Now computed EMD similarity based on "distances" between species contained in `similarity_matrix.csv`)
# This is essentially Pettifor distance, but with non-integer steps defined by data-mining of probabilities.
comp_similarity = gridrdf.earth_mover_distance.composition_similarity_matrix(elem_vectors, 
                                                                     elem_similarity_file = os.path.join(data_source_loc, 'similarity_matrix.csv')
                                                                     )
                                                                     
comp_similarity = comp_similarity.add(comp_similarity.T, fill_value = 0)

total_similarity = 10*grid_similarity + comp_similarity

                                                                     
print('\nStructural Similarity:')
print(grid_similarity)

print('\nComposition Similarity:')
print(comp_similarity)

print('\nTotal similarity (= 10*GRID + Composition):')
print(total_similarity)

                                                                     

