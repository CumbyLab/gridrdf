
import json
import math
import numpy as np
import pandas as pd
from collections import Counter
from pymatgen import Structure, Lattice
from pyemd import emd, emd_with_flow

try:
    from ElMD import ElMD
except:
    print('Element-EMD module not installed')


def emd_example():
    '''
    Test the Earth Mover's Distance (EMD) using similarity matrix 
    against the EMD in the literature
    https://github.com/lrcfmd/ElMD/
    '''
    elem_emd = ElMD()
    comp1 = elem_emd._gen_vector('Li0.7Al0.3Ti1.7P3O12')
    comp2 = elem_emd._gen_vector('La0.57Li0.29TiO3')

    petiffor_emd = elem_emd._EMD(comp1, comp2)

    comp1_reindex = pd.DataFrame(comp1, index=modified_petiffor)
    comp1_reindex = comp1_reindex.reindex(index=petiffor)
    comp2_reindex = pd.DataFrame(comp2, index=modified_petiffor)
    comp2_reindex = comp2_reindex.reindex(index=petiffor)

    dist_matrix = pd.read_csv('similarity_matrix.csv', index_col='ionA').values
    dist_matrix = dist_matrix.copy(order='C')

    em = emd_with_flow(comp1_reindex.values[:,0], comp2_reindex.values[:,0], dist_matrix)
    simi_matrix_emd = em[0]
    emd_flow = pd.DataFrame(em[1], columns=petiffor, index=petiffor)
    emd_flow.replace(0, np.nan).to_csv('emd_flow.csv')


def analysis_emd_100():
    '''
    '''
    df = pd.DataFrame([], index=np.linspace(0.2, 0.6, 5))
    for i in ['small', 'middle', 'large']:
        for thresh in np.linspace(0.2, 0.6, 5):
            data = np.loadtxt(i + '_sample_' + str(thresh), delimiter=' ')
            for val in range(4):
                df.loc[thresh, i + str(val)] = np.count_nonzero(data == val)
    df.to_csv('analysis.csv')


def make_distorted_perovskite():
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

    with open('pero_distortion.json','w') as f:
        json.dump(all_dict, f, indent=1)


def perovskite_different_lattice():
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

    with open('pero_lattice.json','w') as f:
        json.dump(all_dict, f, indent=1)


def dist_matrix_1d(nbin=100):
    '''
    Generate distance matrix for earth mover's distance use
    '''
    dist_matrix = pd.DataFrame()
    for x in range(nbin):
        for y in range(nbin):
            dist_matrix.loc[x, y] = abs(x-y)
    return dist_matrix.values


def nn_bulk_modulus(baseline_id, data, n_nn=1):
    '''
    Using the calucated EMD values and find the n nearest neighbor to estimate 
    bulk modulus

    Args:
        baseline_id: the mp task_id need to be estimated
        data: mp json data from file
        n_nn: number of nearest neighbors
    Return:
        Estimation of the bulk modulus
        Ground truth of bulk modluls
        EMD distance between two structures
    '''
    df_mp = pd.DataFrame.from_dict(data)
    df_mp = df_mp.set_index('task_id')

    df_emd = pd.read_csv(baseline_id + '_emd.csv', index_col=0)
    nn_mps = df_emd.nsmallest(n_nn + 1, baseline_id)
    # sometimes there are other structures have zero emd with 
    # the baseline_id, so it is not always the first row is the baseline_id
    # so use drop instead of removing the first row
    try:
        nn_mps = nn_mps.drop(baseline_id)
    except:
        nn_mps = nn_mps[1:]

    aver_modul = 0
    for task_id in nn_mps.index.values:
        aver_modul += df_mp['elasticity.K_VRH'][task_id]

    return (baseline_id, df_mp['elasticity.K_VRH'][baseline_id], 
            nn_mps.index.values[0], aver_modul / n_nn, 
            nn_mps.values[0][0] )


def rdf_similarity_matrix(data, all_rdf, order=None, method='emd'):
    '''
    Calculate the earth mover's distance between two RDFs
    Current support vanilla rdf and extend rdf
    Using Guassian smearing for rdf

    Args:
        data: data from json
        all_rdf: 
        order: how the compounds are arranged
            None: using the order in the josn 'data'
            symmetry: in the order of space group number, typically for 
                influence of distortion
            lattice: in the order of lattice constant
        method: how similarity is calculated
            emd: earth mover's distance
            linear: inner product
            linear-reciprocal: reciprocal of inner product (sum), useful for visualization
                and comparison with emd results
    Return:
        a pandas dataframe with all pairwise distance
        for multiple shells rdf the distance is the mean value of all shells
    '''
    if len(all_rdf[0].shape) == 2:
        # typically for extend RDF
        df = pd.DataFrame([])
        dist_matrix = dist_matrix_1d(len(all_rdf[0][0]))
        for i1, d1 in enumerate(tqdm(data, desc='rdf similarity', mininterval=60)):
            for i2, d2 in enumerate(data):
                if i1 <= i2:
                    rdf_len = np.array([len(all_rdf[i1]), len(all_rdf[i2])]).min()
                    shell_distances = []
                    for j in range(rdf_len):
                        if method == 'emd':
                            #shell_distances.append(wasserstein_distance(all_rdf[i1][j], all_rdf[i2][j]))
                            shell_distances.append(emd(all_rdf[i1][j], all_rdf[i2][j], dist_matrix))
                        elif method == 'linear' or method == 'linear-reciprocal':
                            shell_distances.append(np.inner(all_rdf[i1][j], all_rdf[i2][j]))

                    if method == 'emd' or method == 'linear':
                        df.loc[d1['task_id'], d2['task_id']] = np.array(shell_distances).mean()
                        df.loc[d2['task_id'], d1['task_id']] = np.array(shell_distances).mean()
                    elif method == 'linear-reciprocal':
                        # add a small number 1e-11 to avoid dividing by zero
                        df.loc[d1['task_id'], d2['task_id']] = 1.0 / (np.array(shell_distances).mean() + 1e-11)
                        df.loc[d2['task_id'], d1['task_id']] = 1.0 / (np.array(shell_distances).mean() + 1e-11)    

    elif len(all_rdf[0].shape) == 1:
        # typically for vanilla RDF
        df = pd.DataFrame([])
        dist_matrix = dist_matrix_1d(len(all_rdf[0]))
        for i1, d1 in enumerate(tqdm(data, desc='rdf similarity', mininterval=60)):
            for i2, d2 in enumerate(data):
                if i1 <= i2:
                    if method == 'emd':
                        #shell_distance = wasserstein_distance(all_rdf[i1], all_rdf[i2])
                        shell_distance = emd(all_rdf[i1], all_rdf[i2], dist_matrix)
                    elif method == 'linear':
                        shell_distance = np.inner(all_rdf[i1], all_rdf[i2])
                    elif method == 'linear-reciprocal':
                        # add a small number 1e-11 to avoid dividing by zero
                        shell_distance = 1.0 / (np.inner(all_rdf[i1], all_rdf[i2]) + 1e-11)
                    df.loc[d1['task_id'], d2['task_id']] = shell_distance
                    df.loc[d2['task_id'], d1['task_id']] = shell_distance    
    
    if order == 'symmetry':
        df2 = pd.DataFrame(columns=['sg'])
        for d in data:
            struct = Structure.from_str(d['cif'], fmt='cif')
            df2.loc[d['task_id']] = struct.get_space_group_info()[1]
        new_index = df2.sort_values('sg').index.values

        df = df.reindex(columns=new_index, index=new_index, fill_value=0)
        np.fill_diagonal(df.values, 0)

    elif order == 'lattice':
        df2 = pd.DataFrame(columns=['lattice'])
        for d in data:
            struct = Structure.from_str(d['cif'], fmt='cif')
            df2.loc[d['task_id']] = struct.lattice.a
        new_index = df2.sort_values('lattice').index.values

        df = df.reindex(columns=new_index, index=new_index, fill_value=0)
        np.fill_diagonal(df.values, 0)
    
    return df 


def rdf_similarity_visualize(data, all_rdf, mode, base_id=31):
    '''
    Visualization of rdf similarity results.
    Currently only used to investigate the influence of lattice constant

    Args:
        data: data from json
        all_rdf: 
        base_id: ID number for the baseline structure, i.e. all others structures are compared 
            to this structure
    Return:
        a pandas dataframe with all pairwise distance
        for multiple shells rdf the distance is the mean value of all shells
    '''

    df = pd.DataFrame([])
    if mode == 'similarity_different_shell':
        vis_rdf_shells = [0, 2, 6, 12, 14, 20, 26, 28]
        for i2, d2 in enumerate(data):
            for j in vis_rdf_shells:
                df.loc[d2['task_id'], str(j)] = wasserstein_distance(all_rdf[base_id][j], all_rdf[i2][j])
    
    elif mode == 'similarity_different_':
        dist_list = []
        mp_indice = []
        vis_ids = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        for i2 in vis_ids:
            mp_indice.append(data[i2]['task_id'])

            rdf_len = np.array([len(all_rdf[base_id]), len(all_rdf[i2])]).min()
            shell_distances = []
            for j in range(rdf_len):
                shell_distances.append(wasserstein_distance(all_rdf[base_id][j], all_rdf[i2][j]))
            dist_list.append(shell_distances)

        df = pd.DataFrame(dist_list, index=mp_indice).transpose()
    
    elif mode == 'rdf_shell':
        rdf0s = []
        local_extrems = [21, 25, 29, 31, 36, 40]
        for i in local_extrems:
            rdf0s.append(all_rdf[i][0])
        df = pd.DataFrame(rdf0s, index=local_extrems).transpose()
    
    elif mode == 'rdf_shell_emd_path':
        emd_flows = []
        local_extrems = [21, 25, 29, 36, 40]
        dist_matrix = dist_matrix_1d(len(all_rdf[0][0]))

        for i in local_extrems:
            em = emd_with_flow(all_rdf[base_id][0], all_rdf[i][0], dist_matrix)
            print(em[0], wasserstein_distance(all_rdf[base_id][0], all_rdf[i][0]))
            emd_flows.append(em[1])
        df = pd.DataFrame(emd_flows, index=local_extrems)
    
    elif mode == 'lattice_difference':
        nums = [0, 259, 544]
        for base_id in nums:
            a1 = Structure.from_str(data[base_id]['cif'], fmt='cif').lattice.a
            for i2, d in enumerate(data):
                a2 = Structure.from_str(data[i2]['cif'], fmt='cif').lattice.a
                rdf_len = np.array([len(all_rdf[base_id]), len(all_rdf[i2])]).min()
                shell_distances = []
                for j in range(rdf_len):
                    shell_distances.append(wasserstein_distance(all_rdf[base_id][j], all_rdf[i2][j]))
                df.loc[data[i2]['task_id'], data[base_id]['task_id'] + 'lattice'] = a1 - a2
                df.loc[data[i2]['task_id'], data[base_id]['task_id']] = np.array(shell_distances).mean()
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Earth Mover Distance',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', type=str, default='../MP_modulus.json',
                        help='the bulk modulus and structure from Materials Project')
    parser.add_argument('--output_file', type=str, default='rdf_similarity',
                        help='currently for rdf similarity')
    parser.add_argument('--rdf_dir', type=str, default='./',
                        help='dir has all the rdf files')
    parser.add_argument('--task', type=str, default='num_shell',
                        help='which property to be calculated: \n' +
                            '   rdf_similarity: \n' +
                            '   rdf_similarity_matrix: \n' +
                            '   rdf_similarity_visualize: \n' +
                            '   '
                      )
    parser.add_argument('--baseline_id', type=str, default='mp-10',
                        help='only used for rdf_similarity task, all other rdfs compared to this')

    if task == 'rdf_similarity':
        rdf_1 = np.loadtxt(rdf_dir + '/' + baseline_id, delimiter=' ')
        df = pd.DataFrame([])
        dist_matrix = dist_matrix_1d(len(rdf_1[0]))

        with open (infile,'r') as f:
            data = json.load(f)

        for d in tqdm(data, desc='rdf similarity', mininterval=60):
            rdf_2 = np.loadtxt(rdf_dir + '/' + d['task_id'], delimiter=' ')
            rdf_len = np.array([len(rdf_1), len(rdf_2)]).min()
            shell_distances = []
            for j in range(rdf_len):
                shell_distances.append(emd(rdf_1[j], rdf_2[j], dist_matrix))
            df.loc[d['task_id'], baseline_id] = np.array(shell_distances).mean()

        df.to_csv(output_file + '_' + baseline_id + '_emd.csv')

    elif task == 'rdf_similarity_matrix':
        with open (infile,'r') as f:
            data = json.load(f)

        all_rdf = rdf_read(data, rdf_dir)
        #all_rdf = rdf_trim(all_rdf, 100)
        #all_rdf = rdf_flatten(all_rdf)
        for similar_measure in ['emd']:
            df = rdf_similarity_matrix(data, all_rdf, method=similar_measure, order=None)
            df.to_csv(output_file + '_' + similar_measure + '_similar_matrix.csv')

    elif task == 'rdf_similarity_visualize':
        with open (infile,'r') as f:
            data = json.load(f)

        all_rdf = rdf_read(data, rdf_dir)

        for mode in ['rdf_shell_emd_path']:
            df = rdf_similarity_visualize(data, all_rdf, mode=mode)
            df.to_csv(output_file + '_' + mode + '.csv')


'''
import os
import json
from earth_mover_distance import nn_bulk_modulus
with open('../MP_modulus.json') as f:
    data = json.load(f)

for f in os.listdir('.'):
    nn_bulk_modulus(f.replace('_emd.csv', ''), data)
'''
