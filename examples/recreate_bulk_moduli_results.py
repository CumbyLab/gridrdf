"""
A script to re-create the bulk modulus results of the original GRID paper.

WARNING: Some aspects of this script may take some time.

"""

__author__ = "James Cumby"
__email__ = "james.cumby@ed.ac.uk"


import gridrdf
import os
import json
import numpy as np
import sklearn


# Step 1 - Import bulk moduli data from materials project
APIkey = os.environ.get('MP_API_KEY')
data_file = 'MP_modulus.json'


if APIkey is None:
    raise ValueError("You need to provide your (legacy) Materials Project API key to run, either as the environment variable `MP_API_KEY` or by manually adding it to this script.")

#print("Extracting bulk modulus data from Materials Project")
#data = gridrdf.data_prepare.get_MP_bulk_modulus_data(APIkey)
#with open(data_file, 'w') as f:
#    json.dump(data, f)
  
# Alternatively, uncomment below to 
# Use the distributed file containing the same
# data as the original manuscript  
with open(data_file, 'r') as f:
    data = json.load(f)

#print(f"Obtained {len(data)} entries with bulk moduli from Materials Project")

# Step 2 - Prepare GRIDs
# Can be commented out if GRIDS already exist somewhere.
print("Calculating GRID representations...")
if os.path.exists('./GRIDS'):
    inp = input('"./GRIDS" folder already exists - do you want to overwrite its contents (y/N)? ')
if len(inp) > 0 and inp.lower()[0] == 'y':
    gridrdf.data_prepare.batch_rdf(data,
                                   max_dist=10,
                                   bin_size = 0.1,
                                   method='kde',
                                   output_dir = './GRIDS',
                                   normalize=True
                                   )
    print("Done.")                               

# Load GRIDS into memory to save time later
#all_GRID = gridrdf.data_io.rdf_read_parallel(data, rdf_dir = './GRIDS/')                              


# Step 3 - Remove any structures with fewer than 100 GRID shells
# Step 4 - remove negative bulk moduli
# Step 5 - filter elements > Bi
# These steps can be run separately (see README) but are combined here using `data_prepare.main`

# Need to set up periodic table definitions first
gridrdf.composition.element_indice()

data_subset = gridrdf.data_prepare.main(data_source = data_file,
                                          tasks = ['subset_composition', 'subset_property', 'subset_grid_len', ],
                                          output_dir = './GRIDS',
                                          MP_API_KEY=APIkey,
                                          output_file = 'subset.json',
                                          num_grid_shells = 100,
                                          composition = {'elem': gridrdf.composition.periodic_table_78, 'type':'consist'},
                                          data_property = ('elasticity.K_VRH', 0, np.inf),
                                          #GRIDS = all_GRID,
                                         )
       

# update GRIDS to match subset by re-reading
print("Re-reading subset of GRIDs")
all_GRID = gridrdf.data_io.rdf_read(data_subset, rdf_dir = './GRIDS/')

# Step 6 - Calculate EMD between all pairs of structures
#### WARNING - THIS MIGHT BE TIME CONSUMING! ####
print(f"Calculating pairwise EMD similarity for {len(data_subset)} materials")
print("This may take some time")
similarity = gridrdf.earth_mover_distance.rdf_similarity_matrix(data_subset, all_GRID, method='emd')
similarity.to_csv('GRID_sim_whole_matrix.csv')

# Only the upper triangle of the matrix is computed, so we need to 
# make it symmetric and convert to NumPy array.
similarity = similarity.add(similarity.T, fill_value=0).to_numpy()

# Step 7 - Use KNN-regressor model to calculate nearest neighbours
# This produces the calculated vs predicted plot reported in the paper.
print("Calculating truth vs prediction data for bulk modulus with K-neighbours regression...")
K_data = np.array([ x['elasticity.K_VRH'] for x in data_subset ])
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1, metric='precomputed')
gridrdf.train.calc_obs_vs_pred_2D(funct = model,
                                 X_data = similarity,
                                 y_data = K_data,
                                 test_size = 0.2,
                                 outdir= './',
                                )
                                
print('Done.')
                                
# Step 8 - Calculate learning curve
print('Calculating learning curve with K-neighbours regression')
gridrdf.train.calc_learning_curve(funct = model,
                                  X_data = similarity, 
                                  y_data = K_data,
                                  test_size=0.1,
                                  procs=1,
                                  output_dir = './')
   
print('Done')
   
 