import unittest
import os
import glob
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.sparse import coo_matrix
from scipy.stats import wasserstein_distance

from gridrdf import earth_mover_distance


FIXTURES_LOC = os.path.join(os.path.dirname(__file__), 'fixtures')

class TestEMD(unittest.TestCase):
    """ Test correct numerical operation of RDF EMD calculation"""
    def setUp(self):
        """
        Set up standard distributions for testing EMD calculations.

        Attributes
        ----------

        self.distances : dict
            Dictionary of arrays of distances
        self.rdf_1D : dict
            1D kernel density estimates for each of the distance sets
        self.rdf_2D : dict
            2D kernel density estimates (i.e. GRID) for each distance set
        self.rdf_2D_sparse : dict
            2D KDE for each distance set, stored as a scipy sparse array
        self.rdf_2D_cumsum : dict
            2D cumulative summation of distance GRIDS

        self.EMD_2D : dict of dicts
            Pairwise EMD values between different distance

        """
        # Set up some simple RDF bin edges
        self.max_dist=10
        self.bin_width = 0.1
        # Decimal place to use for accuracy
        self.decimal = 1
        self.bins = np.linspace(0,self.max_dist, int(self.max_dist/self.bin_width)+1)

        self.distances = []
        self.distances.append(np.array([[1.0,],
                                        [3.0,],
                                        [4.0,],
                                        [4.5,],
                                        [5.0,],
                                        [5.5,],
                                        [6.0,],
                                        [6.5,],
                                        [7.0,],
                                        [7.5,],
                                        ])
                             )
        self.distances.append(np.array([[1.5,],
                                        [3.5,],
                                        [4.5,],
                                        [5.0,],
                                        [5.5,],
                                        [6.0,],
                                        [6.5,],
                                        [7.0,],
                                        [7.5,],
                                        [8.0,],
                                        ])
                             )

        self.distances.append(np.array([[1.2, 1.8,1.8],
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
                              )

        # A simple linear offset makes it easier to know what the correct EMD should be, 
        # as long as the bins do not overlap the edges of the range (which causes anomalies)
        self.distances.append(self.distances[-1] - 0.25)

        # Generate unbroadened 2D histograms
        self.rdf_2D_hist = []
        self.rdf_2D_hist_cumsum = []
        for distances in self.distances:
            dist_hist = [np.histogram(i, bins=self.bins, density=True)[0] for i in distances]
            self.rdf_2D_hist.append(np.array(dist_hist))
            self.rdf_2D_hist_cumsum.append(np.cumsum(self.rdf_2D_hist[-1], axis=-1))


        # Generate 1D KDE distributions manually
        self.rdf_1D = []
        for distances in self.distances:             
            KDE = KernelDensity(kernel='gaussian', bandwidth=0.1)
            log_dens = KDE.fit(distances.ravel()[:, np.newaxis]).score_samples(self.bins[:, np.newaxis])
            dens = np.exp(log_dens)
            self.rdf_1D.append(dens / dens.sum())

        # Generate 2D KDE dists
        self.rdf_2D_kde = []
        self.rdf_2D_sparse = []
        self.rdf_2D_cumsum = []
        for distance_set in self.distances:
            grid = []
            for shell in range(distance_set.shape[0]):
                log_dens = KDE.fit(distance_set[shell][:, np.newaxis]).score_samples(self.bins[:, np.newaxis])
                dens = np.exp(log_dens)
                dens = dens / dens.sum()
                grid.append(dens.copy())

            self.rdf_2D_kde.append(np.array(grid))

            self.rdf_2D_sparse.append(coo_matrix(self.rdf_2D_kde[-1]))

            self.rdf_2D_cumsum.append(np.cumsum(self.rdf_2D_kde[-1], axis=-1))

        # Set up known distances (mean of shells), and fill unknown with NaN
        self.EMD_2D = np.array([[0, 0.5, 2.125, 2.32833333333],
                                [0.5, 0, 2.545, 2.76166666666],
                                [2.125, 2.545, 0, 0.25],
                                [2.328333333333, 2.76166666666, 0.25, 0]])

    def test_pure_wasserstein(self):
        """ Check that wasserstein distance computed directly from distances is OK"""

        for i, dist1 in enumerate(self.distances):
            for j, dist2 in enumerate(self.distances):
                EMD_vals = []
                for shell in range(dist1.shape[0]):
                    EMD_vals.append(wasserstein_distance(dist1[shell], dist2[shell]))
                
                self.assertAlmostEqual(np.mean(EMD_vals), self.EMD_2D[i,j])


    def test_2D_pairwise_optimised_hist(self):
        """ Test optimised _emd_cumsum method with different 2D distributions. """

        for i, dist1 in enumerate(self.rdf_2D_hist_cumsum):
            for j, dist2 in enumerate(self.rdf_2D_hist_cumsum):
                if not np.isnan(self.EMD_2D[i,j]):
                    self.assertAlmostEqual(earth_mover_distance._EMD_cumsum(dist1, dist2, self.bin_width),
                                           self.EMD_2D[i,j],
                                           delta=self.bin_width)

    def test_2D_pairwise_hist(self):
        """ Check rdf_emd_similarity works for 2D case"""

        for i, dist1 in enumerate(self.rdf_2D_hist):
            for j, dist2 in enumerate(self.rdf_2D_hist):
                if not np.isnan(self.EMD_2D[i,j]):
                    # Check that the histograms are correct accounting for the bin_width
                    self.assertAlmostEqual(earth_mover_distance.rdf_emd_similarity(dist1, dist2, self.max_dist, method='orig'),
                                           self.EMD_2D[i,j],
                                           delta=self.bin_width)

                    
    def test_2D_pairwise_kde(self):
        """ Check rdf_emd_similarity works for 2D case"""

        for i, dist1 in enumerate(self.rdf_2D_kde):
            for j, dist2 in enumerate(self.rdf_2D_kde):
                if not np.isnan(self.EMD_2D[i,j]):
                    # Check that the EMDs are correct accounting for the bin_width
                    self.assertAlmostEqual(earth_mover_distance.rdf_emd_similarity(dist1, dist2, self.max_dist, method='orig'),
                                           self.EMD_2D[i,j],
                                           delta=self.bin_width)

    def test_2D_optimised_row(self):
        """ Check _emd_cumsum_row gives correct values. """

        EMDs = earth_mover_distance._emd_cumsum_row(np.array(self.rdf_2D_cumsum), 0, self.bin_width)
        # Test to the second decimal place
        np.testing.assert_array_almost_equal(EMDs, self.EMD_2D[0], decimal=self.decimal)

    def test_emd_duplicated(self):
        """ Test emd with the same RDF returns 0. """
        self.assertEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_1D[0], self.rdf_1D[0]), 0)
        
    def test_emd_1D_simple(self):
        self.assertAlmostEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_1D[0], self.rdf_1D[1], max_distance=self.max_dist), 0.5)

    def test_emd_1D_complex(self):
        self.assertAlmostEqual(earth_mover_distance.rdf_emd_similarity(self.rdf_1D[2], self.rdf_1D[3], max_distance=self.max_dist), 0.25)

    def test_emd_2D_method(self):
        """ Check original and faster 2D methods give same answer for complex case"""
        orig_method = earth_mover_distance.rdf_emd_similarity(self.rdf_2D_kde[2], self.rdf_2D_kde[3], max_distance=self.max_dist, method='orig')
        fast_method = earth_mover_distance.rdf_emd_similarity(self.rdf_2D_kde[2], self.rdf_2D_kde[3], max_distance=self.max_dist, method='fast')

        self.assertAlmostEqual(orig_method, fast_method)

    def test_emd_2D_sparse(self):
        """ Check fast and sparse 2D methods give same answer"""
        sparse_method = earth_mover_distance.rdf_emd_similarity(self.rdf_2D_sparse[2], self.rdf_2D_sparse[3], max_distance=self.max_dist, method='fast')
        fast_method = earth_mover_distance.rdf_emd_similarity(self.rdf_2D_kde[2], self.rdf_2D_kde[3], max_distance=self.max_dist, method='fast')

        self.assertAlmostEqual(sparse_method, fast_method)        

    def test_emd_matrix(self):
        test_dists = earth_mover_distance.super_fast_EMD_matrix(self.rdf_2D_hist,
                                                                self.bin_width,
                                                                weighting='constant',)
        
        np.testing.assert_almost_equal(test_dists, self.EMD_2D, decimal=self.decimal)

        
class TestHelperFunctions(unittest.TestCase):

    def test_dist_matrix(self):
        dim = 10
        dist_mat = earth_mover_distance.dist_matrix_1d(dim)
        self.assertEquals(dist_mat.shape, (dim,dim))

        test_mat = np.abs(np.meshgrid(range(dim), range(dim))[0] - np.meshgrid(range(dim), range(dim))[1])

        np.testing.assert_allclose(dist_mat, test_mat)    
        
if __name__ == '__main__':
    unittest.main()