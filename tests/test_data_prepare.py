import unittest
import os
import glob

from gridrdf import data_prepare

from tqdm import tqdm
from functools import partialmethod

# Hack to turn off tqdm output during tests
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

FIXTURES_LOC = os.path.join(os.path.dirname(__file__), 'fixtures')

class TestNaCl(unittest.TestCase):
    def setUp(self):
        self.nacl = data_prepare.nacl()
    def test_type(self):
        self.assertIsInstance(self.nacl, data_prepare.Structure)
    def test_lattice(self):
        self.assertEqual(self.nacl.lattice.a, 5.6)
        self.assertTrue(self.nacl.lattice.is_orthogonal)
    def test_atoms(self):
        self.assertEqual(len(self.nacl), 8)
        
class TestDistortedPerovskite(unittest.TestCase):
    def setUp(self):
        self.pero = data_prepare.make_distorted_perovskite(outfile=None)
    def test_length(self):
        self.assertEqual(len(self.pero), 6)
    def test_keys(self):
        self.assertIn('cif', self.pero[0])
        
class TestPerovskiteLattice(unittest.TestCase):
    def setUp(self):
        self.pero = data_prepare.perovskite_different_lattice(outfile=None)
    def test_length(self):
        self.assertEqual(len(self.pero), 61)
        
        
class TestBatchRDF(unittest.TestCase):
    """ Test that batch_RDF correctly produces files"""
    def setUp(self):
        os.mkdir('./_test_fixtures')
        self.data = data_prepare.make_distorted_perovskite(outfile=None)
        data_prepare.batch_rdf(self.data, output_dir = './_test_fixtures')
        self.files = glob.glob('./_test_fixtures/*')
    def test_count(self):
        self.assertEqual(len(self.files), len(self.data))
    def tearDown(self):
        for file in self.files:
            os.remove(file)
        os.rmdir('./_test_fixtures')
        
        
class TestMainOnline(unittest.TestCase):
    """ Check operation of main routine using online perovskite generation. """
    def setUp(self):
        self.data = data_prepare.make_distorted_perovskite(outfile=None)
    def test_composition(self):
        data = data_prepare.main(data_source = 'perovskite_distort',
                          tasks = ['subset_composition',],
                          composition = {'elem':['Ti'], 'type': 'exclude'},
                          output_file = None,
                          )
        self.assertEqual(len(data), 0)
    def test_spacegroup(self):
        data = data_prepare.main(data_source = 'perovskite_distort',
                          tasks = ['subset_space_group',],
                          space_groups = range(1,11,1),
                          output_file = None,
                          )
                          
        # Exact symmetry depends on PyMatGen tolerances, but 
        # cubic should be filtered away.
        self.assertTrue(len(data) < len(self.data))  

    def test_grid_missing(self):
        """ Test correct operation when GRIDS have not been calculated """
        
        with self.assertRaises(OSError):
            data = data_prepare.main(data_source = 'perovskite_distort',
                                     tasks = ['subset_grid_len',],
                                     output_dir = './_test_fixtures',
                                     num_grid_shells = 100)

    def test_grid_bin(self):
        data = data_prepare.main(data_source = 'perovskite_distort',
                          tasks = ['grid_rdf_bin',],
                          output_file = None,
                          output_dir = './_test_fixtures',
                          max_dist = 10,
                          )
                          
        self.assertEqual(len(data), len(glob.glob('./_test_fixtures/*')))
                                     
    def test_grid_kde(self):
        data = data_prepare.main(data_source = 'perovskite_distort',
                          tasks = ['grid_rdf_kde',],
                          output_file = None,
                          output_dir = './_test_fixtures',
                          max_dist = 10,
                          )
                          
        self.assertEqual(len(data), len(glob.glob('./_test_fixtures/*')))
        
    def test_basic_rdf(self):
        data = data_prepare.main(data_source = 'perovskite_distort',
                          tasks = ['original_rdf',],
                          output_file = None,
                          output_dir = './_test_fixtures',
                          max_dist = 10,
                          )
                          
        self.assertEqual(len(data), len(glob.glob('./_test_fixtures/*')))
                  
    def test_multitask(self):  
        data = data_prepare.main(data_source = 'perovskite_distort',
                          tasks = ['grid_rdf_kde', 'subset_grid_len'],
                          output_file = None,
                          output_dir = './_test_fixtures',
                          max_dist = 6,
                          num_grid_shells = 80
                          )
                          
        self.assertEqual(len(data), len(glob.glob('./_test_fixtures/*')))                                               
                                     
    def tearDown(self):
        if os.path.isdir('./_test_fixtures'):
            for file in glob.glob('./_test_fixtures/*'):
                os.remove(file)
            os.rmdir('./_test_fixtures')
        
class TestMainOffline(unittest.TestCase):
    """ Check operation of main routine using test files """
    def setUp(self):
        self.pero_json = os.path.join(FIXTURES_LOC, 'pero_lattice.json')
        self.pero_grids = os.path.join(FIXTURES_LOC, 'pero_grids')
        self.orig_len = 61
    def test_rdf_len(self):
        data = data_prepare.main(data_source = self.pero_json,
                          tasks = ['subset_grid_len',],
                          output_dir = self.pero_grids,
                          num_grid_shells = 100,
                          )
        self.assertEqual(len(data), 13)

        
if __name__ == '__main__':
    unittest.main()