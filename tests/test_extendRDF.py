import unittest
import os
import glob

from gridrdf import extendRDF
from gridrdf import data_prepare

from pymatgen.core.structure import Structure

from tqdm import tqdm
from functools import partialmethod

# Hack to turn off tqdm output during tests
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

FIXTURES_LOC = os.path.join(os.path.dirname(__file__), 'fixtures')

class TestPairwiseDistances(unittest.TestCase):
    def setUp(self):
        self.nacl = data_prepare.nacl()
        self.low_symm = Structure.from_file(os.path.join(FIXTURES_LOC, 'Hf5CuSn3.cif'))
        self.frac_occ = Structure([[4,0,0],[0,4,0],[0,0,4]], [{'Fe':0.4, 'Mn':0.5}], [[0,0,0]])

    def test_num_sites(self):
        """ Check distance and num_neighbour cutoffs give the same answer in NaCl. """
        neighbours_cutoff = extendRDF.get_pairwise_distances(self.nacl, cutoff = 3)
        neighbours_count = extendRDF.get_pairwise_distances(self.nacl, num_neighbours=6)
        self.assertEqual(len(neighbours_cutoff), 8)
        self.assertEqual(len(neighbours_count), 8)
        self.assertCountEqual(neighbours_count[0], neighbours_cutoff[0])
        self.assertListEqual(neighbours_cutoff, neighbours_count)
    def test_sorted_distances(self):
        """ Check neighbours are sorted by increasing distance. """
        neighbours = extendRDF._sorted_neighbours(self.nacl, 3.0)
        self.assertEqual(len(neighbours), 8)
        for site in neighbours:
            current_dist = 0
            for n in site:
                self.assertGreaterEqual(n.nn_distance, current_dist)
                current_dist = n.nn_distance
    def test_neighbour_cutoffs(self):
        """ Check cutoffs are adequate given a simple geometry.
        
        Note that these values are likely to be most inaccurate for small cutoffs/neighbours, but the 
        returned values should be large enough to work correctly.

        """

        # Check distances give sensible numbers of neighbours
        self.assertGreaterEqual(extendRDF._estimate_cutoff(self.nacl, 6), self.nacl.lattice.a/2)
        self.assertGreaterEqual(extendRDF._estimate_cutoff(self.nacl, 26), self.nacl.lattice.a)

        # Check number of predicted neighbours for a given cutoff
        self.assertLessEqual(extendRDF._estimate_neighbours(self.nacl, self.nacl.lattice.a/2), 6)

        # Check that `get_pairwise_distances` correctly updates cutoff
        neighbours, new_cutoff = extendRDF.get_pairwise_distances(self.nacl, num_neighbours=26, return_cutoff=True)
        self.assertLessEqual(new_cutoff, self.nacl.lattice.a *1.0009)

    def test_neighour_finding(self):
        """ Check that neighbours are found correctly. """

        # Specifying num_neighbours and cutoff together should raise an error
        self.assertRaises(AssertionError, extendRDF.get_pairwise_distances, self.nacl, num_neighbours=50, cutoff=5)

        # Neighbours should have the correct shape
        neighbours = extendRDF.get_pairwise_distances(self.nacl, num_neighbours=50)
        self.assertEqual(len(neighbours), len(self.nacl))
        self.assertListEqual([len(i) for i in neighbours], [50]*len(self.nacl))

        pymatgen_neighbours = self.low_symm.get_all_neighbors(3.5)
        pymatgen_neighbour_count = [len(i) for i in pymatgen_neighbours]
        low_symm_neighbours = extendRDF.get_pairwise_distances(self.low_symm, cutoff=3.5)
        
        # A fixed cutoff should return the same result as PyMatGen
        self.assertListEqual(pymatgen_neighbour_count, [len(j) for j in low_symm_neighbours])

        # A fixed number of neighbours should give equal sizes, but an increased cutoff
        fixed_neighbours = max(pymatgen_neighbour_count)
        low_symm_neighbours, cutoff = extendRDF.get_pairwise_distances(self.low_symm, num_neighbours = fixed_neighbours, return_cutoff=True)
        self.assertListEqual([len(i) for i in low_symm_neighbours], [fixed_neighbours]*len(self.low_symm))

        self.assertGreaterEqual(cutoff, 3.5)

    def test_occupancies(self):
        """ Check occupancies are extracted correctly. """

        neighbours = self.frac_occ.get_all_neighbors(30)

        dists, occs = extendRDF._neighbours_to_dists(neighbours)

        self.assertListEqual(list(dists.shape), list(occs.shape))
        # Check all elements of the partially occupied structure have total 0.9
        self.assertEqual(occs.min(), 0.9)
        self.assertEqual(occs.max(), 0.9)



if __name__ == '__main__':
    unittest.main()