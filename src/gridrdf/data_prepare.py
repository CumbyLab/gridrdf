"""
Interface to collate or generate input data.

Helper functions to simplify extraction of data from the Materials Project
or to generate 'standard' datasets (such as artificial perovskites).
"""

# Bash script to generate RDF and GRID for structures
'''
for rdf in smooth_rdf vanilla_rdf ; do cd $rdf ; \
for file in pero_distortion pero_lattice rp_srtio3 ; \
do python ../../../descriptors/data_explore.py --input_file ../$file.json \
 --task rdf_similarity  --output_file ../${file}_$rdf ; \
done ; cd .. ; done 
'''

import json
import math
import argparse
import time
import numpy as np
import pandas as pd
import os
import warnings

import tqdm
from collections import Counter
try:
    from pymatgen import Structure
except ImportError:
    from pymatgen.core.structure import Structure   
try:
    from pymatgen import Lattice
except ImportError:
    from pymatgen.core.lattice import Lattice

from pymatgen.ext.matproj import MPRester

from . import extendRDF
from gridrdf.composition import elements_selection
from gridrdf.otherRDFs import origin_rdf_histo
from gridrdf.data_io import rdf_read, rdf_read_parallel


def nacl():
    '''
    Generate NaCl structure, usually for test purpose

    Args:
        None
    Return:
        a pymatgen structure of NaCl
    '''
    nacl = Structure.from_spacegroup(
        'Fm-3m', Lattice.cubic(5.6),['Na', 'Cl'],[[0.5, 0.5, 0.5], [0, 0, 0]] )
    return nacl


def get_MP_bulk_modulus_data(APIkey): 
    '''
    Get all Material project structures with calculated elastic properties

    Args:
        APIkey: a string 'Hecxxxxxxxxxxx' needed to be applied from MP website
    Return:
        A json object containing
            MP task id
            Bulk modulus
            Shear modulus
            Elastic anisotropy
            CIF of structure
    '''
    # put personal API key here
    q = MPRester(APIkey)   
    print(f"Using Materials Project database v{q.get_database_version()}")
    
    d = q.query(criteria={'elasticity.K_VRH':{'$nin': [None, '']}}, 
                properties=['task_id', 
                            'elasticity.K_VRH', 
                            'elasticity.G_VRH', 
                            'elasticity.elastic_anisotropy', 
                            'cif'])
    
    
    return d


def get_ICSD_CIFs_from_MP(APIkey): 
    '''
    Get all ICSD structures in the Material project 

    Args:
        APIkey: a string 'Hecxxxxxxxxxxx' needed to be applied from MP website
    Return:
        A json object containing
            MP task id
            ICSD id
            CIF of structure
    '''
    q = MPRester(APIkey)   
    d = q.query(criteria={'icsd_ids':{'$nin': [None, []]}}, 
                properties=['task_id', 'icsd_ids', 'cif'])
    return d

def _insert_field(infile1='num_shell', infile2='MP_modulus_all.json', 
                outfile='MP_modulus_v4.json'):
    '''
    Insert new file in the json file

    Args:
        infile1: file containing values to be inserted, the field should
            consult data_explore.py
        infile2: file into which new field will be inserted
        outfile: a new file contains new inserted fields
    Return:
        None
    '''
    results = np.loadtxt(infile1, delimiter=' ')
    with open(infile2, 'r') as f:
        data = json.load(f)

    for i, d in enumerate(data):
        d['average_bond_length'] = results[i][2]
        d['bond_length_std'] = results[i][3]    

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=1)
    
    return


def _json_order():
    '''
    Make the perovskite structure in the order of lattice constant
    '''

    with open('pero2.json','r') as f:
        data = json.load(f)

    l = []
    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        l.append(struct.lattice.a)

    df2 = pd.DataFrame(l, columns=['lattice'])
    new_index = df2.sort_values('lattice').index.values

    d2 = []
    for i in new_index:
        d2.append(data[i])

    with open('pero3.json','w') as f:
        json.dump(d2,f,indent=1)

def make_distorted_perovskite(outfile = None):
    '''
    used for testing purpose for EMD + extended RDF as a similarity measure

    '''
    cif = "# generated using pymatgen\ndata_SrTiO3\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.94513000\n_cell_length_b   3.94513000\n_cell_length_c   3.94513000\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   90.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   SrTiO3\n_chemical_formula_sum   'Sr1 Ti1 O3'\n_cell_volume   61.40220340\n_cell_formula_units_Z   1\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Sr  Sr0  1  0.00000000  0.00000000  0.00000000  1\n  Ti  Ti1  1  0.50000000  0.50000000  0.50000000  1\n  O  O2  1  0.50000000  0.00000000  0.50000000  1\n  O  O3  1  0.50000000  0.50000000  0.00000000  1\n  O  O4  1  0.00000000  0.50000000  0.50000000  1\n"
    s = Structure.from_str(cif, fmt='cif')

    posits = [
            [0.50, 0.50, 0.50],
            [0.50, 0.50, 0.51],
            [0.50, 0.50, 0.52],
            [0.50, 0.50, 0.53],
            [0.50, 0.507, 0.507],
            [0.506, 0.506, 0.506]
        ]

    all_dict = []
    for i, posit in enumerate(posits):
        one_dict = {}
        one_dict['task_id'] = 'pero_distort_' + str(i)
        s[1] = 'Ti', posit
        one_dict['cif'] = s.to(fmt='cif')
        all_dict.append(one_dict)

    if outfile is not None:
        with open(outfile,'w') as f:
            json.dump(all_dict, f, indent=1)

    return all_dict

def perovskite_different_lattice(outfile=None):
    cif = "# generated using pymatgen\ndata_SrTiO3\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.94513000\n_cell_length_b   3.94513000\n_cell_length_c   3.94513000\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   90.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   SrTiO3\n_chemical_formula_sum   'Sr1 Ti1 O3'\n_cell_volume   61.40220340\n_cell_formula_units_Z   1\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Sr  Sr0  1  0.00000000  0.00000000  0.00000000  1\n  Ti  Ti1  1  0.50000000  0.50000000  0.50000000  1\n  O  O2  1  0.50000000  0.00000000  0.50000000  1\n  O  O3  1  0.50000000  0.50000000  0.00000000  1\n  O  O4  1  0.00000000  0.50000000  0.50000000  1\n"
    s = Structure.from_str(cif, fmt='cif')

    all_dict = []
    for lat in np.linspace(3, 6, 61):
        one_dict = {}
        one_dict['task_id'] = 'pero_latt_' + str(round(lat,3))
        new_latt = Lattice.from_parameters(lat, lat, lat, 90, 90, 90)
        s.lattice = new_latt
        one_dict['cif'] = s.to(fmt='cif')
        all_dict.append(one_dict)

    if outfile is not None:
        with open(outfile,'w') as f:
            json.dump(all_dict, f, indent=1)
            
    return all_dict
            
def batch_rdf(data,
              max_dist=10,
              num_neighbours = 100,
              bin_width=0.1,
              broadening=0.1,
              normalize=True,
              output_dir = './',
              ):
    '''
    Read structures and output the GRID representations to files

    Parameters
    ----------

    data: iterable
        List of dicts containing (as a minimum) the 'task_id' and 'cif' keys. Intended to 
        be generated from Materials Project.
    max_dist: float, optional
        Maximum distance to compute shells to. Alternatively, specify "num_neighbours" directly 
        to avoid truncating shells later.
    num_neighbours : int, optional
        Number of nearest neighbours to consider in RDF calculation (alternatively, specify max_dist).
    broadening : float, default 0.1
        Gaussian broadening to apply to RDF representation. If 0, histogram will be 
        created without broadening.
        For more control over the broadening process, see `extendRDF.calculate_RDF`.
    normalize : Bool, default True
        Whether to normalize the resulting histogram area.

        
    Notes
    -----

    If both max_dist and num_neighbours are specified, GRID will be calculated on the basis of 
    num_neighbours.
    '''

    structures = []
    # First, generate PyMatGen objects from CIF representations
    for d in data:
        structures.append(Structure.from_str(d['cif'], fmt='cif'))

    orig_max_dist = max_dist
    # Determine whether to use max_dist or num_neighbours
    if num_neighbours and max_dist:
        max_dist = None

    # Generate all neighbour lists for all structures to the same cutoff
    neighbours, cutoffs = extendRDF.find_all_neighbours(structures,
                                               num_neighbours=num_neighbours,
                                               cutoff=max_dist,
                                               return_limits = True,
                                               dryrun = False)
    
    max_dist = max(np.round(cutoffs[1], 1), orig_max_dist)
    if max_dist != orig_max_dist:
        print(f"Maximum distance has been updated to {max_dist} to account for {num_neighbours} neighbours")


    rdf_results = []

    # Compute RDF representation
    for i, struct in tqdm.tqdm(enumerate(structures)):
        rdf_results.append(extendRDF.calculate_rdf(struct,
                                                   neighbours[i],
                                                   rdf_type='grid',
                                                   max_dist=max_dist,
                                                   bin_width=bin_width,
                                                   smearing = broadening,
                                                   normed = normalize,
                                                   broadening_method = 'convolve',
                                                   return_sparse = False)
                            )

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Save GRIDs to individual output files.
    warnings.warn("GRIDs will be saved in a different format in a future version.", FutureWarning)
    for i, GRID in enumerate(rdf_results):
        outfile = os.path.normpath(os.path.join(output_dir, data[i]['task_id']))
        np.savetxt(outfile, GRID, delimiter=' ', fmt='%.6f')
    
def trim_rdf_bins(data,
                  all_rdf,
                  number_of_shells,
                  write_to_disk = True,
                  output_dir = './',
                  disk_time = 0.01,
                  ):
    """ Truncate all RDFs to the largest number of bins possible in the group. 
    
    Args:
        data: input data from Materials Project
        all_rdf: list of np.arrays containing RDF (or GRID) information
        number_of_shells: number of RDF shells to trim all_GRID entries to (uses
                          first RDF axis by default)
        write_to_disk: whether to update existing RDF files on disk based on results
                       (default False). This will update GRIDs that are trimmed, but 
                       does NOT remove GRIDs that are shorter than the required length.
        outdir: directory to write (or update) RDF files to
        gzip_file: in case RDF files take too much disk space, set 
                    this to true to gzip the files (not yet test)
        disk_time: time to allow between disk writes (default 0.01)
        
    Returns:
        num_trimmed: The number of RDF entries that are above the cutoff length and were
                     trimmed to match the cutoff.
        num_removed: The number of RDF entries that were below the cutoff and were removed
                     from `data`.
    """
    
    assert number_of_shells > 0
    max_shells = np.max([i.shape[0] for i in all_rdf])
    assert number_of_shells <= max_shells, f"Number of shells chosen ({number_of_shells}) is bigger than the largest number calculated ({max_shells})"
    
    num_removed = 0
    num_trimmed = 0
    new_data = []
    new_rdfs = []
    
    for i, d in enumerate(data):
        if all_rdf[i].shape[0] < number_of_shells:
            num_removed += 1
            continue
                
        elif all_rdf[i].shape[0] > number_of_shells:
            #print('Started: ', all_rdf[i].shape[0])
            all_rdf[i] = all_rdf[i][:number_of_shells]
            num_trimmed += 1
            #print('Ended: ', all_rdf[i].shape[0])
            
            if write_to_disk:
                outfile = os.path.normpath(os.path.join(output_dir, d['task_id']))

                np.savetxt(outfile, all_rdf[i][:number_of_shells], delimiter=' ', fmt='%.6f')
                time.sleep(disk_time)
            
                
        new_data.append(data[i])
        new_rdfs.append(all_rdf[i])
                
    return new_data, new_rdfs, num_trimmed, num_removed
    
    
def main(data_source = 'MP_modulus.json',
         tasks = ['grid_rdf_kde',],
         MP_API_KEY = None,
         composition = {'elem':[], 'type':'consist'},
         output_dir = './outputs/',
         output_file = None,
         space_groups = [],
         data_property = (None, -np.inf, np.inf),
         num_grid_shells = 100,
         max_dist = 10,
         GRIDS = None,
        ):
    """ Main logic of data preparation. 
    
    
    Args:
        data_source: str
            Source of MP data to process. Possible options are:
                - nacl
                    - data are generated using `nacl` function
                - perovskite_lattice
                    - generated using `perovskite_different_lattice`
                - perovskite_distort
                    - generated using `make_distorted_perovskite`
                - None or MP_bulk_modulus
                    - data containing bulk moduli are extracted from MP.
                - ends with '.json'
                    - data are assumed to be in a json-formatted file
        tasks: list of str
            One or more tasks to be performed based on the input data in order supplied.
            Options are:
                - subset_composition
                    Filter based on elements. See required 'composition' argument.
                - subset_grid_len
                    Filter any structures containing fewer than `min_grid_groups`
                    GRID groups for the given distance cutoff (requires GRID files 
                    to have been generated first, either as a separate task or by 
                    specifying a suitable `output_dir`).    
                - subset_space_group
                    Filter structures by spacegroup number(s) as a list
                - subset_property
                    Filter by materials having a specific property (defined within the data source)
                - grid_rdf_bin
                    Compute GRID with basic histogram binning
                - grid_rdf_kde
                    Compute GRID with Gaussian smoothing of distances
                - original_rdf
                    Compute basic RDF
        MP_API_KEY: str
            Materials project API key 
        composition: dict
            If task == subset_composition, this dict is required. Keys are 
                'elem': list of species to be passed to `composition.elements_selection`
                'type': method to filter structures: 'include' (greedy AND), 'exclude' (NOT)
                        or 'consist' (ONLY)
        output_dir : str
            Location to save outputted file(s), for instance GRIDs.
        output_file : str
            File name for filtered data (if required).
        space_groups : list of ints
            If task == 'subset_space_group', this argument is the space group numbers
            to be retained. If len(space_groups) == 0, all space groups will be kept.
        data_property : list
            Property contained in data_source (e.g. 'elasticity.K_VRH') and the (min, max) 
            values it should adopt (inclusive)
        max_dist : float
            Maximum distance to calculate RDFs up to.
        num_grid_shells : int
            Number of GRID groups to trim/filter data to
                        
            
                    
    """

    assert 'type' in composition
    assert 'elem' in composition
    
    if data_source.lower() == 'nacl':
        data = nacl()
    elif data_source.lower() == 'perovskite_lattice':
        data = perovskite_different_lattice()
    elif data_source.lower() == 'perovskite_distort':
        data = make_distorted_perovskite()
    elif data_source.endswith('.json'):
        with open(data_source, 'r') as f:
            data = json.load(f)
    elif data_source is None or data_source.lower() == 'mp_bulk_modulus':
        if MP_API_KEY is None:
            raise ValueError('To access the materials project API, an API key is required')
        data = get_MP_bulk_modulus_data(MP_API_KEY)
    else:
        raise ValueError(f'Unknown data source {data_source}')
        
    original_length = len(data)
        
    
    for task in tasks:  
        print("Performing ", task, "on ", len(data), "entries.")
        if task == 'subset_composition':
            #print(elem_list)
            data = elements_selection(data, elem_list=composition['elem'], mode=composition['type'])

                
        elif task == 'subset_space_group':
            if len(space_groups) == 0:
                sg_kept = range(230, 0, -1)
            else:
                sg_kept = [int(i) for i in space_groups]
 
            for d in data:
                struct = Structure.from_str(d['cif'], fmt='cif')
                sg_num = struct.get_space_group_info()[1]
                if sg_num not in sg_kept:
                    data.remove(d)

        elif task == 'subset_grid_len':
            if GRIDS is None:
                all_rdf = rdf_read_parallel(data, output_dir)
            else:
                all_rdf = GRIDS
                
            data, all_rdf, trimmed, removed = trim_rdf_bins(data = data,
                                                             all_rdf = all_rdf,
                                                             number_of_shells = num_grid_shells,
                                                             write_to_disk = True,
                                                             output_dir = output_dir,
                                                             )
                                                             
            #GRIDS = all_rdf
            
            print(f"    Removed {removed} entries and trimmed {trimmed} entries based on a cutoff of {num_grid_shells} shells")


        elif task == 'subset_property':
            key = data_property[0]
            key_min = data_property[1]
            key_max = data_property[2]
            for d in data[:]:
                if d[key] < key_min or d[key] > key_max:
                    data.remove(d)
                    
        elif task == 'grid_rdf_bin':
            # Compute un-broadened GRID
            batch_rdf(data, max_dist=max_dist, num_neighbours = num_grid_shells, broadening=0.0, output_dir = output_dir)
        elif task == 'grid_rdf_kde':
            #Compute 0.1Ang broadened GRID
            batch_rdf(data, max_dist=max_dist, num_neighbours = num_grid_shells, broadening=0.1, output_dir = output_dir)
        elif task == 'original_rdf':
            origin_rdf_histo(data, max_dist=max_dist, output_dir = output_dir)

        else:
            raise ValueError(f'Unknown task {task}')
            
        print("Removed {} structure entries based on {}".format(original_length - len(data), task))
        original_length = len(data)

        
        
    # Save subset to new output file if needed
    if output_file is not None:
        outf = os.path.normpath(os.path.join(output_dir, output_file))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(outf, 'w') as f:
            json.dump(data, f, indent=1)
            
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset prepare',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_source', type=str, default='../MP_modulus.json',
                        help='the source of structure data in materials project format, as json file.')
    parser.add_argument('-f', '--output_file', type=str, default='subset.json',
                        help='outpu')
    parser.add_argument('-d', '--output_dir', type=str, default='./',
                        help='Directory to output data file(s) to')
    parser.add_argument('--tasks', nargs='+', type=str, default='',
                        help='Which task(s) to perform on the dataset: \n' +
                        
                            '   subset_composition: select a subset which have specified elements: \n' +
                            '   subset_grid_len: drop all the ext-RDF with a length less than 100 \n' +
                            '   subset_space_group: restrict to specified spacegroup number(s)\n' +
                            '   grid_rdf_bin: compute GRID for all data_sources with simple binning\n' +
                            '   grid_rdf_kde: compute GRID for all data_sources with Gaussian distance broadening \n' + 
                            '   original_rdf: compute basic RDF for all data_sources'
                      )
                      
    parser.add_argument('--elem_list', nargs = '+',     type=str, default='O',
                        help='only used for subset task')
    parser.add_argument('--elem_method', type=str, default='include',
                        help='method to use for composition filtering (include, exclude or consist)')
    parser.add_argument('--spacegroups', nargs='+', type=str, default='',
                        help='spacegroup numbers to filter by (can include inclusive ranges, i.e. 40-56)')
    parser.add_argument('--max_dist', type=float, default=10.0,
                        help='Cutoff distance of the RDF')       
    parser.add_argument('--min_grid_groups', type=int, default=100,
                        help = 'Minimum number of GRID groups required, below which data will be omitted if task us subset_grid_len.')
    parser.add_argument('--prop_filter', nargs=3, metavar=('KEY', 'MIN', 'MAX'), default=['elasticity.K_VRH', '0', 'inf'],
                        help = 'Property (e.g. `elasticity.K_VRH`) contained in data_source, and the min/max values it can take.')

    args = parser.parse_args()

    comp_dict = {'type': args.elem_method, 'elem': args.elem_list}
    
    data_prop = []
    data_prop.append(args.prop_filter[0])
    for val in args.prop_filter[1:]:
        val = val.strip("'").strip('"')
        if 'inf' in val:
            if val[0] == '-':
                data_prop.append(-np.inf)
            else:
                data_prop.append(np.inf)
        else:
            data_prop.append(float(val))
        
    spacegroups = []
    for sg in args.spacegroups:
        if '-' in sg:
            start = int(sg.split('-')[0])
            end = int(sg.split('-')[1])
            spacegroups += range(start, end+1, 1)
        else:
            spacegroups.append(int(sg))
            
    
    
    data = main(data_source = args.data_source,
                tasks = args.tasks,
                composition = comp_dict,
                output_file = args.output_file,
                output_dir = args.output_dir,
                max_dist = args.max_dist,  
                num_grid_shells = args.min_grid_groups,
                data_property = data_prop,
                )
                
    print('There are {} items in data'.format(len(data)))