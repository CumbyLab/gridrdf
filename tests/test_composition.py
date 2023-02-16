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

class TestComposition(unittest.TestCase):
    def setUp(self):
        self.nacl = data_prepare.nacl()
        self.low_symm = Structure.from_file(os.path.join(FIXTURES_LOC, 'Hf5CuSn3.cif'))
        self.frac_occ = Structure([[4,0,0],[0,4,0],[0,0,4]], [{'Fe':0.4, 'Mn':0.5}], [[0,0,0]])

    def test_deuterium(self):
        """ Check that composition works correctly for deuterium. """

        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()