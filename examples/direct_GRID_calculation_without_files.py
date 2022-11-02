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

script_loc = os.path.dirname(__file__)
data_source_loc = os.path.join(script_loc, '../data_sources')

# Variables to define cutoffs etc
maximum_grid_distance = 10
bin_size = 0.1
number_of_shells = 100

# Set up dummy pymatgen Structures with different cells/compositions
dummy_structures = [
    Structure(Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60), ["Si", "Si"], [[0, 0, 0], [0.75, 0.5, 0.75]]),
    Structure(Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60), ["Ni", "Ni"], [[0, 0, 0], [0.75, 0.5, 0.75]]),
    Structure(Lattice.from_parameters(a=3.94, b=3.94, c=3.94, alpha=120, beta=90, gamma=60), ["Si", "Si"], [[0, 0, 0], [0.75, 0.5, 0.75]]),
]

# Empty list to hold GRID arrays
grid_representations = []


# Calculate all GRIDS iteratively
for structure in dummy_structures:
    prim_cell_list = list(range(len(structure)))
    rdf_atoms = gridrdf.extendRDF.get_rdf_and_atoms(structure=structure, prim_cell_list=prim_cell_list, 
                                                    max_dist=maximum_grid_distance)

    GRID = gridrdf.extendRDF.rdf_kde(rdf_atoms=rdf_atoms, max_dist=maximum_grid_distance, bin_size=bin_size)
    
    assert GRID.shape[0] >= number_of_shells, f"Distance cutoff should be increased so that there are at least {number_of_shells} GRID shells for {structure} (only {GRID.shape[0]} computed)."
    
    grid_representations.append(GRID[:number_of_shells])
    
# Calculate EMD similarity
# Currently, rdf_similarity_matrix requires a list of dicts, each with a 'task_id' for each structure
# NOTE - this is likely to change in a future release
structure_ids = [ {'task_id': i, 'structure':dummy_structures[i]} for i in range(len(grid_representations)) ]

# Calculate EMD similarity between grids
grid_similarity = gridrdf.earth_mover_distance.rdf_similarity_matrix(structure_ids, grid_representations, method='emd')    

grid_similarity = grid_similarity.add(grid_similarity.T, fill_value=0)

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

                                                                     

