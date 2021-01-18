
from pymatgen import Structure
cif = "# generated using pymatgen\ndata_SrTiO3\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.94513000\n_cell_length_b   3.94513000\n_cell_length_c   3.94513000\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   90.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   SrTiO3\n_chemical_formula_sum   'Sr1 Ti1 O3'\n_cell_volume   61.40220340\n_cell_formula_units_Z   1\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Sr  Sr0  1  0.00000000  0.00000000  0.00000000  1\n  Ti  Ti1  1  0.50000000  0.50000000  0.50000000  1\n  O  O2  1  0.50000000  0.00000000  0.50000000  1\n  O  O3  1  0.50000000  0.50000000  0.00000000  1\n  O  O4  1  0.00000000  0.50000000  0.50000000  1\n"
s = Structure.from_str(cif, fmt='cif')

posits = [[0.50, 0.50, 0.50],
        [0.50, 0.50, 0.51],
        [0.50, 0.50, 0.52],
        [0.50, 0.50, 0.53],
        [0.50, 0.51, 0.51],
        [0.51, 0.51, 0.51]]

all_dict = []
for i, posit in enumerate(posits):
    one_dict = {}
    one_dict['task_id'] = 'generate-' + str(i)
    s[1] = 'Ti', posit
    one_dict['cif'] = s.to(fmt='cif')
    all_dict.append(one_dict)

with open('pero_gene.json','w') as f:
    json.dump(all_dict, f, indent=1)

