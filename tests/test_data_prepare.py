import unittest

from gridrdf import data_prepare

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
        
if __name__ == '__main__':
    unittest.main()