.. 
	These are examples of badges you might want to add to your README:
	   please update the URLs accordingly

	.. image:: https://api.cirrus-ci.com/github/<USER>/gridrdf.svg?branch=main
		 :alt: Built Status
		 :target: https://cirrus-ci.com/github/<USER>/gridrdf
	.. image:: https://readthedocs.org/projects/gridrdf/badge/?version=latest
		 :alt: ReadTheDocs
		 :target: https://gridrdf.readthedocs.io/en/stable/
	.. image:: https://img.shields.io/coveralls/github/<USER>/gridrdf/main.svg
		 :alt: Coveralls
		 :target: https://coveralls.io/r/<USER>/gridrdf

	.. image:: https://img.shields.io/conda/vn/conda-forge/gridrdf.svg
		 :alt: Conda-Forge
		 :target: https://anaconda.org/conda-forge/gridrdf

.. image:: https://img.shields.io/pypi/v/gridrdf.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/gridrdf/

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

.. image:: https://zenodo.org/badge/515119558.svg
   :alt: Latest Zenodo badge
   :target: https://zenodo.org/badge/latestdoi/515119558

=======
gridrdf
=======


    Grouped representation of interatomic distances (GRID)


This package is designed to compute GRID descriptions of 
crystal structures and use them to train ML models, currently
based on properties extracted from the Materials Project.
In addition, it contains a number of tools for computing 
earth mover's distance (EMD) between distributions such 
as GRID or RDF, and using the resulting dissimilarities for
further calculations.

This code accompanies the following paper, which should be cited
if you use it for any future publications:

`Grouped Representation of Interatomic Distances as a Similarity Measure for Crystal Structures <https://doi.org/10.26434/chemrxiv-2022-9m4jh>`_




------------
Installation
------------

The latest stable version of gridrdf can be installed using pip:

.. code-block:: bash

    pip install gridrdf


If you are using conda, you may find it easier to create a new environment with the
required dependencies first, before installing gridrdf using pip:

.. code-block:: bash

	conda create -n gridrdf_env -f environment.yml
	conda activate gridrdf_env
	pip install gridrdf


Alternatively, the most recent development version can be installed
by cloning the git repository, and then installing in 'development' mode:

.. code-block:: bash

	git clone https://git.ecdf.ed.ac.uk/funcmatgroup/gridrdf.git
	pip install -e gridrdf/

Using conda with this approach, you can install the dependencies from requirements.txt:

.. code-block:: bash

	git clone https://git.ecdf.ed.ac.uk/funcmatgroup/gridrdf.git
	conda env create -n gridrdf_env --file gridrdf/requirements.txt -c defaults -c conda-forge
	conda activate gridrdf_env
	pip install -e gridrdf


-------
Testing
-------

Once downloaded or installed, it is recommended to test the code operates
correctly. Using a python terminal, navigate to the `gridrdf` directory and type

.. code-block:: bash

	python -m unittest discover -s tests

--------------
Using the Code
--------------

All modules contained in gridrdf have documentation describing their
intended use, and are grouped into 'data preparation' (`gridrdf.data_prepare`),
'similarity calculation' (`gridrdf.earth_mover_distance`) and 'model training' (`gridrdf.train`) steps. 
Other utility modules are also included.

Submodules of gridrdf can be imported and used interactively in a python environment, but the main steps
outlined above can also be accessed as command line scripts by calling the module directly (--help will give 
more details of usage):

.. code-block:: bash

	python -m gridrdf.MODULE_NAME --help


-----------------
Intended Workflow
-----------------

To re-create the results presented in the publication of predicting
bulk modulus  using a kNN model and EMD dissimilarity, the procedure is as follows:

1. Import data from the materials project with calculated elastic moduli
	
    .. code-block:: python
	
	    data = gridrdf.data_prepare.get_MP_bulk_modulus_data(APIkey)
	    with open('MP_modulus.json') as f:
			gridrdf.json.dumps(data, f)
   
    NOTE: gridrdf currently relies on the legacy Materials Project API, so needs an old API KEY
   
2. Calculate GRID representation for each structure (generates GRID file for each structure)
    .. code-block:: python
	
		gridrdf.data_prepare.batch_rdf(data[:2],
									   max_dist=10,
									   bin_size = 0.1,
									   method='kde',
									   output_dir = './GRIDS',
									   normalize=True
									  )

    or from a terminal:
   
    .. code-block:: bash
	
		python -m gridrdf.data_prepare --data_source MP_modulus.json --output_dir ../GRIDS/ --tasks grid_rdf_kde

   
3. Remove any structures with fewer than 100 GRID shells
    .. code-block:: python
	
		all_GRID = gridrdf.data_io.rdf_read_parallel(data, rdf_dir = './GRIDS/')
		for i, d in enumerate(data[:]):
			if len(all_GRID[i]) < 100:
				data.remove(d)
		with open('MP_subset.json', 'w') as f:
			json.dump(data, f, indent=1)
 
   or from a terminal:
    .. code-block:: bash

		python -m gridrdf.data_prepare --data_source MP_modulus.json --output_dir ./GRIDS/ --tasks subset_grid_len --output_file MP_subset.json  

    
4. Filter structure with negative bulk moduli
	.. code-block:: python
	
		for d in data:
			if d['elasticity.K_VRH'] < 0:
				data.remove(d)

   or from a terminal:
	.. code-block:: bash
   
		python -m gridrdf.data_prepare --data_source MP_modulus.json --output_dir ./GRIDS/ --output_file MP_subset.json --tasks subset_property --prop_filter elasticity.K_VRH 0 np.inf

   
5. Filter elements with atomic number > Bi:
	.. code-block:: python
	
		# First, generate internal list of 78 elements (as gridrdf.composition.periodic_table_78)
		gridrdf.composition.element_indice()
		data = gridrdf.data_prepare.elements_selection(data, gridrdf.composition.periodic_table_78, mode='consist')

   
   NOTE: not currently implemented for command line script
    
Steps 2-5 can be combined into a single function call (similarly through terminal script by specifying tasks in order):

.. code-block:: python

	data_quick = gridrdf.data_prepare.main(data_source = './MP_modulus.json',
									  tasks = ['subset_grid_len', 'subset_composition', 'subset_property'],
									  output_dir = './GRIDS',
									  output_file = 'subset.json',
									  max_dist=10,
									  min_grid_groups = 100,
									  composition = {'elem': gridrdf.composition.periodic_table_78, 'type':'consist'},
									  data_property = ('elasticity.K_VRH', 0, np.inf)
									 )
    
    
6. Calculate pair-wise dissimilarity matrix between structures using EMD (time-consuming)
	.. code-block:: python
	
		similarity = gridrdf.earth_mover_distance.rdf_similarity_matrix(data, all_GRID, method='emd')
		similarity.to_csv('GRID_sim_whole_matrix.csv')

   or from a terminal:
    .. code-block:: bash
	
		python -m gridrdf.earth_mover_distance --input_file MP_modulus.json --rdf_dir ./GRIDS/ --output_file GRID_sim --task rdf_similarity_matrix

   Note: The data can also be processed in smaller chunks using `indice` (or `--data_indice` as a script) to allow parallel-processing.
7. Use a simplified kNN model to predict bulk modulus
	.. code-block:: python
	
		K_data = np.array([ x['elasticity.K_VRH'] for x in data ])
		model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1, metric='precomputed')
		gridrdf.train.calc_obs_vs_pred_2D(funct = model,
										 X_data = similarity,
										 y_data = K_data,
										 test_size = 0.2,
										 outdir= './',
										)

   or from a terminal:
	.. code-block:: bash
	
		python -m gridrdf.train --input_file MP_modulus.json --rdf_dir ./GRIDS/ --input_features distance_matrix --dist_matrix GRID_sim_whole_matrix.csv --out_dir ./ --funct knn_reg --target bulk_modulus --metrics emd --task obs_vs_pred

   
   
------
Issues
------

If you have any questions, comments or problems with the code, please feel free to post them as issues `here <https://git.ecdf.ed.ac.uk/funcmatgroup/gridrdf/-/issues>`_ 
   


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
