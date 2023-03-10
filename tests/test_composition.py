import unittest
import os
import glob
import numpy as np

import gridrdf

from pymatgen.core.structure import Structure

from tqdm import tqdm
from functools import partialmethod

# Hack to turn off tqdm output during tests
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

FIXTURES_LOC = os.path.join(os.path.dirname(__file__), 'fixtures')

class TestComposition(unittest.TestCase):
    def setUp(self):
        self.nacl = gridrdf.data_prepare.nacl()
        self.low_symm = Structure.from_file(os.path.join(FIXTURES_LOC, 'Hf5CuSn3.cif'))
        self.frac_occ = Structure([[4,0,0],[0,4,0],[0,0,4]], [{'Fe':0.4, 'Mn':0.5}], [[0,0,0]])
        self.deuterated = Structure(4*np.eye(3), ['D','O'], [[0,0,0],[0.5,0.5,0.5]])

    def test_deuterium(self):
        """ Check that composition works correctly for deuterium. """

        neighbours = gridrdf.extendRDF.get_pairwise_distances(self.deuterated, num_neighbours=10)
        comp = gridrdf.composition.composition_hist(self.deuterated, neighbours=neighbours)
        
        self.assertEqual(comp.sum(axis=0)[1], comp.shape[0] / 2)


if __name__ == '__main__':
    unittest.main()