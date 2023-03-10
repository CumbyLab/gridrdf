"""
A script to re-create the bulk modulus results of the original GRID paper.

WARNING: Some aspects of this script may take some time.

The slowest aspect is calculating EMD dissimilarity between all pairs. Currently, computing the full EMD matrix (~12,000**2 pairs)
takes around 30 minutes on an intel i5-8500 (3.00 GHz, 6-cores) processor, using efficient parallelisation.

"""

__author__ = "James Cumby"
__email__ = "james.cumby@ed.ac.uk"


import gridrdf
import os
import json
import numpy as np
import sklearn

script_location = os.path.dirname(__file__)

results_location = os.path.join(script_location, 'testing')
if not os.path.exists(results_location):
    os.makedirs(results_location)

data_source_loc = os.path.join(script_location, '../data_sources/')
    

# File to use for initial structure data
# If MP_API_KEY variable is not set, this script will try to read
# data in from the file specified, otherwise it will write out the 
# materials extracted from the Materials Project.

# This is the file containing all bulk-moduli containing structures
#data_file_basename = 'MP_modulus.json'    

# This is a smaller 'test set' to use as an example.
data_file_basename = 'small_test_set.json'

# Step 1 - Import bulk moduli data from materials project
APIkey = os.environ.get('MP_API_KEY')
if APIkey is None:
    print("You need to provide your (legacy) Materials Project API key as the environment variable `MP_API_KEY` to use the most recent data from Materials Project.")
    print(f"\nReverting to use the included file `data_sources/{data_file_basename}`")
    data_file = os.path.join(data_source_loc, data_file_basename)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
else:
    data_file = os.path.join(results_location, data_file_basename)
    print(f"Extracting bulk modulus data from Materials Project and saving to {data_file}" )
    
    data = gridrdf.data_prepare.get_MP_bulk_modulus_data(APIkey)
    with open(data_file, 'w') as f:
        json.dump(data, f)
    
print(f"Obtained {len(data)} entries with bulk moduli.")


GRID_location = os.path.join(results_location, 'GRIDS')

# Step 2 - Prepare GRIDs

if os.path.exists(GRID_location):
    inp = input(f'Folder {GRID_location} already exists - do you want to overwrite its contents (y/N)? ')
else:
    inp = 'y'

print("Calculating GRID representations...")    
if len(inp) > 0 and inp.lower()[0] == 'y':
    gridrdf.data_prepare.batch_rdf(data,
                                   max_dist=10,
                                   bin_width = 0.1,
                                   broadening=0.1,
                                   output_dir = GRID_location,
                                   normalize=True
                                   )
    print("Calculating Grids - Done.")                               
                         

# Step 3 - Remove any structures with fewer than 100 GRID shells
# Step 4 - remove negative bulk moduli
# Step 5 - filter elements > Bi
# These steps can be run separately (see README) but are combined here using `data_prepare.main`

# Need to set up periodic table definitions first
gridrdf.composition.element_indice()

data_subset = gridrdf.data_prepare.main(data_source = data_file,
                                          tasks = ['subset_composition', 'subset_property', 'subset_grid_len', ],
                                          output_dir = GRID_location,
                                          MP_API_KEY=APIkey,
                                          output_file = 'subset.json',
                                          num_grid_shells = 100,
                                          composition = {'elem': gridrdf.composition.periodic_table_78, 'type':'consist'},
                                          data_property = ('elasticity.K_VRH', 0, np.inf),
                                          #GRIDS = all_GRID,
                                         )
       

# update GRIDS to match subset by re-reading
print("Re-reading subset of GRIDs")
all_GRID = gridrdf.data_io.rdf_read(data_subset, rdf_dir = GRID_location)

# Step 6 - Calculate EMD between all pairs of structures
#### WARNING - THIS MIGHT BE TIME CONSUMING! ####
print(f"Calculating pairwise EMD similarity for {len(data_subset)} materials")
print("This may take some time")
similarity = gridrdf.earth_mover_distance.rdf_similarity_matrix(all_GRID, data_subset, method='emd')
similarity.to_csv(os.path.join(results_location, 'GRID_similarity_matrix.csv'))


# Step 7 - Calculate composition similarity
print("Calculating composition similarity")

# First, convert composition to vector encoding
elem_vectors, elem_symbols = gridrdf.composition.composition_one_hot(data_subset, method='percentage')

# Now computed EMD similarity based on "distances" between species contained in `similarity_matrix.csv`)
# This is essentially Pettifor distance, but with non-integer steps defined by data-mining of probabilities.
comp_similarity = gridrdf.earth_mover_distance.composition_similarity_matrix(elem_vectors, 
                                                                     elem_similarity_file = os.path.join(data_source_loc, 'similarity_matrix.csv')
                                                                     )
comp_similarity.to_csv(os.path.join(results_location, 'composition_similarity_matrix.csv'))
                                                                     

total_similarity = 10*similarity + comp_similarity
                                                                     
# Only the upper triangle of the matrix is computed, so we need to 
# make it symmetric and convert to NumPy array.
total_similarity = total_similarity.add(total_similarity.T, fill_value=0)
total_similarity.to_csv(os.path.join(results_location, 'total_similarity_matrix.csv'))

total_similarity = total_similarity.to_numpy()

# Step 7 - Use KNN-regressor model to calculate nearest neighbours
# This produces the calculated vs predicted plot reported in the paper.
print("Calculating truth vs prediction data for bulk modulus with K-neighbours regression...")
K_data = np.array([ x['elasticity.K_VRH'] for x in data_subset ])
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1, metric='precomputed')
gridrdf.train.calc_obs_vs_pred_2D(funct = model,
                                 X_data = total_similarity,
                                 y_data = K_data,
                                 test_size = 0.2,
                                 outdir= results_location,
                                )
                                
print('Done.')
                                
# Step 8 - Calculate learning curve
print('Calculating learning curve with K-neighbours regression')
gridrdf.train.calc_learning_curve(funct = model,
                                  X_data = total_similarity, 
                                  y_data = K_data,
                                  test_size=0.1,
                                  procs=1,
                                  output_dir = results_location,
                                  )
   
print('Done')
   
 