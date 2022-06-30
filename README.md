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




# Installation

These files can either be imported as a python package (`gridrdf`) by 
adding the `gridrdf` directory to PYTHON_PATH, or used as commandline
scripts by using the `python -m gridrdf.MODULE` mechanism.

The package can also be installed in development mode using `pip` or 
`conda`.

# Testing

Once downloaded or installed, it is recommended to test the code operates
correctly. Usibng a python terminal, navigate to the `gridrdf` directory and type

``` bash
python -m unittest discover -s tests
```

# Using the Code

All modules contained in gridrdf have documentation describing their
intended use. Note that many modules can also be run as command-line 
scripts; to get more details type:
```
python -m gridrdf.MODULE_NAME --help
```

To re-create the results of kNN model to predict bulk modulus
using the GRID descriptor and EMD as the dissimilarity measure,
the procedure is as follows:

1. Import data from the materials project with calculated
   elastic moduli
   ``` python
   data = get_MP_bulk_modulus_data(APIkey)
   with open('MP_modulus.json') as f:
       gridrdf.json.dumps(data, f)
   ```
2. Calculate GRID representation for each structure (generates GRID file for each structure)
   ``` python
	gridrdf.data_prepare.batch_rdf(data[:2],
								   max_dist=10,
								   bin_size = 0.1,
								   method='kde',
								   output_dir = './GRIDS',
								   normalize=True
								  )
   ```
   or from a terminal:
   
   ``` bash
   python -m gridrdf.data_prepare --data_source MP_modulus.json --output_dir ../GRIDS/ --tasks grid_rdf_kde
   ```
   
3. Remove any structures with fewer than 100 GRID shells
   ```python
   all_GRID = gridrdf.dataio.rdf_read_parallel(data, rdf_dir = './GRIDS/')
   for i, d in enumerate(data[:]):
       if len(all_GRID[i]) < 100:
           data.remove(d)
   with open('MP_subset.json', 'w') as f:
       json.dump(data, f, indent=1)
   ```   
   or from a terminal:
   ``` bash
   python -m gridrdf.data_prepare --data_source MP_modulus.json --output_dir ./GRIDS/ --tasks subset_grid_len --output_file MP_subset.json  
   ```
    
4. Filter structure with negative bulk moduli
   ``` python
   for d in data:
       if d['elasticity.K_VRH'] < 0:
           data.remove(d)
   ```
   or from a terminal:
   ``` bash
   python -m gridrdf.data_prepare --data_source MP_modulus.json --output_dir ./GRIDS/ --output_file MP_subset.json --tasks subset_property --prop_filter elasticity.K_VRH 0 np.inf
   ```
   
5. Filter elements with atomic number > Bi:
   ``` python
   # First, generate internal list of 78 elements (as gridrdf.composition.periodic_table_78)
   gridrdf.composition.element_indice()
   data = gridrdf.data_prepare.elements_selection(data, gridrdf.composition.periodic_table_78, mode='consist')
   ```
   
   NOTE: not currently implemented for command line script
	
These data preparation tasks can be combined into a single function call (similarly through terminal script):

``` python
data_quick = gridrdf.data_prepare.main(data_source = './MP_modulus.json',
                                  tasks = ['subset_grid_len', 'subset_composition', 'subset_property'],
                                  output_dir = './GRIDS',
                                  output_file = 'subset.json',
                                  max_dist=10,
                                  min_grid_groups = 100,
                                  composition = {'elem': gridrdf.composition.periodic_table_78, 'type':'consist'},
                                  data_property = ('elasticity.K_VRH', 0, np.inf)
                                 )
```
	
	
6. Calculate pair-wise dissimilarity matrix between structures using EMD (time-consuming)
   ```
   similarity = gridrdf.earth_mover_distance.rdf_similarity_matrix(data, all_GRID, method='emd')
   similarity.to_csv('GRID_sim_whole_matrix.csv')
   ```
   or from a terminal:
   ```
   python earth_mover_distance.py --input_file MP_modulus.json --rdf_dir ./GRIDS/ --output_file GRID_sim --task rdf_similarity_matrix
   ```
   Note: The data can also be processed in smaller chunks using `indice` (or `--data_indice` as a script) to allow parallel-processing.
7. Use a simplified kNN model to predict bulk modulus
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
   
