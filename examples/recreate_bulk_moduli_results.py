"""
A script to re-create the bulk modulus results of the original GRID paper.

WARNING: Some aspects of this script may take some time.

"""

__author__ = "James Cumby"
__email__ = "james.cumby@ed.ac.uk"


import gridrdf


# Step 1 - Import bulk moduli data from materials project
APIkey = None
if APIkey is None:
    raise ValueError("You need to provide your Materials Project API key to run")

data = gridrdf.get_bulk_modulus_data(APIkey)
with open('MP_modulus.json') as f:
    gridrdf.json.dumps(data, f)

# Step 2 - Prepare GRIDs
gridrdf.data_prepare.batch_rdf(data[:2],
                               max_dist=10,
                               bin_size = 0.1,
                               method='kde',
                               output_dir = './GRIDS',
                               normalize=True
                              )
                              
# Step 3 - Remove any structures with fewer than 100 GRID shells
# Step 4 - remove negative bulk moduli
# Step 5 - filter elements > Bi
# These steps can be run separately (see README) but are combined here

data_subset = gridrdf.data_prepare.main(data_source = './MP_modulus.json',
                                  tasks = ['subset_grid_len', 'subset_composition', 'subset_property'],
                                  output_dir = './GRIDS',
                                  output_file = 'subset.json',
                                  max_dist=10,
                                  min_grid_groups = 100,
                                  composition = {'elem': gridrdf.composition.periodic_table_78, 'type':'consist'},
                                  data_property = ('elasticity.K_VRH', 0, np.inf)
                                 )
       
# Updated json file with new subset
with open('MP_subset.json', 'w') as f:
   json.dump(data_subset, f, indent=1)     

# Step 6 - Calculate EMD between all pairs of structures
#### WARNING - THIS MIGHT BE TIME CONSUMING! ####
similarity = gridrdf.earth_mover_distance.rdf_similarity_matrix(data, all_GRID, method='emd')
similarity.to_csv('GRID_sim_whole_matrix.csv')


# Step 7 - Use KNN-regressor model to calculate nearest neighbours
# This produces the calculated vs predicted plot reported in the paper.
K_data = np.array([ x['elasticity.K_VRH'] for x in data ])
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1, metric='precomputed')
gridrdf.train.calc_obs_vs_pred_2D(funct = model,
                                 X_data = similarity,
                                 y_data = K_data,
                                 test_size = 0.2,
                                 outdir= './',
                                )
   
   
 