import unittest
import os
import glob
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.sparse import coo_matrix

from gridrdf import earth_mover_distance


FIXTURES_LOC = os.path.join(os.path.dirname(__file__), 'fixtures')

class TestEMD(unittest.TestCase):
    """ Test correct numerical operation of RDF EMD calculation"""
    def setUp(self):
        # Set up some simple RDF bin edges
        self.max_dist=5
        self.bin_width = 0.1
        self.bins = np.linspace(0,self.max_dist, int(self.max_dist/self.bin_width)+1)

        self.distances = {}
        self.distances['simple_a'] = np.array([[1.0, 2.0],
                                               [3.0, 3.0]])
        self.distances['simple_b'] = np.array([[1.5, 2.5],
                                               [3.5, 3.5]])

        self.distances['complex_a'] = np.array([[1.2, 1.8,1.8],
                                                [1.8, 2.1, 2.2],
                                                [1.95, 2.6, 2.6],
                                                [2.4, 2.9, 2.7],
                                                [2.6, 3.5, 3.4],
                                                [3.1, 3.8, 3.8],
                                                [2.1, 2.6, 2.7],
                                                [3.3, 3.9, 3.9],
                                                [3.7, 4.2, 4.2],
                                                [4.1, 4.5, 4.4]
                                                ])

        # A simple linear offset makes it easier to know what the correct EMD should be, 
        # as long as the bins do not overlap the edges of the range (which causes anomalies)
        self.distances['complex_b'] = self.distances['complex_a'] - 0.25


        # Generate 1D KDE distributions manually
        self.rdf_1D = {}
        for distance_set in self.distances:             
            KDE = KernelDensity(kernel='gaussian', bandwidth=0.1)
            log_dens = KDE.fit(self.distances[distance_set].ravel()[:, np.newaxis]).score_samples(self.bins[:, np.newaxis])
            dens = np.exp(log_dens)
            self.rdf_1D[distance_set] = dens / dens.sum()

        # Generate 2D KDE dists
        self.rdf_2D = {}
        self.rdf_2D_sparse = {}
        for distance_set in self.distances:
            grid = []
            for shell in range(self.distances[distance_set].shape[0]):
                log_dens = KDE.fit(self.distances[distance_set][shell][:, np.newaxis]).score_samples(self.bins[:, np.newaxis])
                dens = np.exp(log_dens)
                dens = dens / dens.sum()
                grid.append(dens.copy())

            self.rdf_2D[distance_set] = np.array(grid)

            self.rdf_2D_sparse[distance_set] = coo_matrix(self.rdf_2D[distance_set])

    def test_emd_duplicated(self):
        """ Test emd with the same RDF returns 0. """
        self.assertEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_1D['simple_a'], self.rdf_1D['simple_a']), 0)
        self.assertEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_2D['simple_a'], self.rdf_2D['simple_a']), 0)
        
    def test_emd_1D_simple(self):
        self.assertAlmostEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_1D['simple_a'], self.rdf_1D['simple_b'], max_distance=self.max_dist), 0.5)

    def test_emd_1D_complex(self):
        self.assertAlmostEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_1D['complex_a'], self.rdf_1D['complex_b'], max_distance=self.max_dist), 0.25)

    def test_emd_2D_simple(self):
        self.assertAlmostEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_2D['simple_a'], self.rdf_2D['simple_b'], max_distance=self.max_dist), 0.5)

    def test_emd_2D_complex(self):
        self.assertAlmostEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_2D['complex_a'], self.rdf_2D['complex_b'], max_distance=self.max_dist), 0.25)

    def test_emd_2D_method(self):
        """ Check original and faster 2D methods give same answer for complex case"""
        orig_method = earth_mover_distance.rdf_emd_similarity(self.rdf_2D['complex_a'], self.rdf_2D['complex_b'], max_distance=self.max_dist, method='orig')
        fast_method = earth_mover_distance.rdf_emd_similarity(self.rdf_2D['complex_a'], self.rdf_2D['complex_b'], max_distance=self.max_dist, method='fast')

        self.assertAlmostEqual(orig_method, fast_method)

    def test_emd_2D_sparse(self):
        """ Check fast and sparse 2D methods give same answer"""
        sparse_method = earth_mover_distance.rdf_emd_similarity(self.rdf_2D_sparse['complex_a'], self.rdf_2D_sparse['complex_b'], max_distance=self.max_dist, method='fast')
        fast_method = earth_mover_distance.rdf_emd_similarity(self.rdf_2D['complex_a'], self.rdf_2D['complex_b'], max_distance=self.max_dist, method='fast')

        self.assertAlmostEqual(sparse_method, fast_method)        

        

        
if __name__ == '__main__':
    unittest.main()