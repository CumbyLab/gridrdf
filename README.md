# Grouped Representation of Interatomic Distances (GRID)

This package is designed to compute GRID descriptions of 
crystal structures and use them to train ML models, currently
based on properties extracted from the Materials Project.
In addition, it contains a number of tools for computing 
earth mover's distance (EMD) between distributions such 
as GRID or RDF, and using the resulting disimilarities for
further calculations.

This code accompanies the following paper, which should be cited
if you use it for any future publications:

CITATION HERE WHEN ACCEPTED


# Using the Code

All modules contained in gridrdf have documentation describing their
intended use. Note that many modules can also be run as command-line 
scripts; to get more details type:
```
python MODULE_NAME.py --help
```


To re-create the results of kNN model to predict bulk modulus
using the GRID descriptor and EMD as the dissimilarity measure,
the procedure is as follows:

1. Import data from the materials project with calculated
   elastic moduli
   ``` python
   data = gridrdf.data_prepare.get_MP_bulk_modulus_data(APIkey)
   with open('MP_modulus.json') as f:
       gridrdf.json.dumps(data, f)
   ```
2. Calculate GRID representation for each structure (generates GRID file for each structure)
   ``` python
   gridrdf.data_explore.batch_rdf(data,
                                  max_dist = 10,
                                  bin_size = 0.1,
                                  method='kde',
                                  output_dir = './GRIDS/',
                                  normalize = True,
                                  )
   ```
   or from a terminal:
   ```
   python data_explore.py --input_file MP_modulus.json --rdf_dir ./GRIDS/ --task extend_rdf_kde
   ```
3. Remove any structures with negative bulk modulus and truncate to 100 GRID shells
   ```python
   all_GRID = gridrdf.dataio.rdf_read(data, rdf_dir = './GRIDS/')
   for i, d in enumerate(data[:]):
       if len(all_GRID[i]) < 100:
           data.remove(d)
   with open('MP_subset.json', 'w') as f:
       json.dump(data, f, indent=1)
   ```   
   or from a terminal:
   ```
   python data_prepare.py --input_file MP_modulus.json --rdf_dir ./GRIDS/ --output_file MP_subset.json --task subset_rdf_len
   ```
    
	*****************************************************
	** HOW WERE DATA FILTERED BY MODULUS and ELEMENTS? **
	*****************************************************
	
4. Calculate pair-wise dissimilarity matrix between structures using EMD (time-consuming)
   ```
   similarity = gridrdf.earth_mover_distance.rdf_similarity_matrix(data, all_GRID, method='emd')
   similarity.to_csv('GRID_sim_whole_matrix.csv')
   ```
   or from a terminal:
   ```
   python earth_mover_distance.py --input_file MP_modulus.json --rdf_dir ./GRIDS/ --output_file GRID_sim --task rdf_similarity_matrix
   ```
   Note: The data can also be processed in smaller chunks using `indice` (or `--data_indice` as a script) to allow parallel-processing.
5. Use a simplified kNN model to predict bulk modulus
   ```
   K_data = np.array([ x['elasticity.K_VRH'] for x in data ])
   model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1, metric='precomputed')
   gridrdf.train.calc_obs_vs_pred_2D(funct = model,
                                     X_data = similarity,
                                     y_data = K_data,
                                     test_size = 0.2,
                                     outdir= './',
                                    )
   ```
   or from a terminal:
   ```
   python train.py --input_file MP_modulus.json --rdf_dir ./GRIDS/ --input_features distance_matrix --dist_matrix GRID_sim_whole_matrix.csv --out_dir ./ --funct knn_reg --target bulk_modulus --metrics emd --task obs_vs_pred
   ```
   
