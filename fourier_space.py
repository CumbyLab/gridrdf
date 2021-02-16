# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

'''
Modified from pymatgen/analysis/diffraction/xrd.py
https://pymatgen.org/pymatgen.analysis.diffraction.xrd.html
'''

import os
import json
import argparse
from math import sin, cos, asin, pi, degrees, radians
import numpy as np

from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.core import DiffractionPattern, AbstractDiffractionPatternCalculator, \
    get_unique_families


# XRD wavelengths in angstroms
WAVELENGTHS = {
    'CuKa': 1.54184,
    'CuKa2': 1.54439,
    'CuKa1': 1.54056,
    'CuKb1': 1.39222,
    'MoKa': 0.71073,
    'MoKa2': 0.71359,
    'MoKa1': 0.70930,
    'MoKb1': 0.63229,
    'CrKa': 2.29100,
    'CrKa2': 2.29361,
    'CrKa1': 2.28970,
    'CrKb1': 2.08487,
    'FeKa': 1.93735,
    'FeKa2': 1.93998,
    'FeKa1': 1.93604,
    'FeKb1': 1.75661,
    'CoKa': 1.79026,
    'CoKa2': 1.79285,
    'CoKa1': 1.78896,
    'CoKb1': 1.63079,
    'AgKa': 0.560885,
    'AgKa2': 0.563813,
    'AgKa1': 0.559421,
    'AgKb1': 0.497082,
}

ATOMIC_SCATTERING_PARAMS = {
    'Ru': [[4.358, 27.881], [3.298, 5.179], [1.323, 0.594], [0, 0]],
    'Re': [[5.695, 28.968], [4.74, 5.156], [2.064, 0.575], [0, 0]],
    'Ra': [[6.215, 28.382], [5.17, 5.002], [2.316, 0.562], [0, 0]],
    'Rb': [[4.776, 140.782], [3.859, 18.991], [2.234, 3.701], [0.868, 0.419]],
    'Rn': [[4.078, 38.406], [4.978, 11.02], [3.096, 2.355], [1.326, 0.299]],
    'Rh': [[4.431, 27.911], [3.343, 5.153], [1.345, 0.592], [0, 0]],
    'Be': [[1.25, 60.804], [1.334, 18.591], [0.36, 3.653], [0.106, 0.416]],
    'Ba': [[7.821, 117.657], [6.004, 18.778], [3.28, 3.263], [1.103, 0.376]],
    'Bi': [[3.841, 50.261], [4.679, 11.999], [3.192, 2.56], [1.363, 0.318]],
    'Bk': [[6.502, 28.375], [5.478, 4.975], [2.51, 0.561], [0, 0]],
    'Br': [[2.166, 33.899], [2.904, 10.497], [1.395, 2.041], [0.589, 0.307]],
    'H':  [[0.202, 30.868], [0.244, 8.544], [0.082, 1.273], [0, 0]],
    'D':  [[0.202, 30.868], [0.244, 8.544], [0.082, 1.273], [0, 0]],
    'P':  [[1.888, 44.876], [2.469, 13.538], [0.805, 2.642], [0.32, 0.361]],
    'Os': [[5.75, 28.933], [4.773, 5.139], [2.079, 0.573], [0, 0]],
    'Ge': [[2.447, 55.893], [2.702, 14.393], [1.616, 2.446], [0.601, 0.342]],
    'Gd': [[5.225, 29.158], [4.314, 5.259], [1.827, 0.586], [0, 0]],
    'Ga': [[2.321, 65.602], [2.486, 15.458], [1.688, 2.581], [0.599, 0.351]],
    'Pr': [[5.085, 28.588], [4.043, 5.143], [1.684, 0.581], [0, 0]],
    'Pt': [[5.803, 29.016], [4.87, 5.15], [2.127, 0.572], [0, 0]],
    'Pu': [[6.415, 28.836], [5.419, 5.022], [2.449, 0.561], [0, 0]],
    'C':  [[0.731, 36.995], [1.195, 11.297], [0.456, 2.814], [0.125, 0.346]],
    'Pb': [[3.51, 52.914], [4.552, 11.884], [3.154, 2.571], [1.359, 0.321]],
    'Pa': [[6.306, 28.688], [5.303, 5.026], [2.386, 0.561], [0, 0]],
    'Pd': [[4.436, 28.67], [3.454, 5.269], [1.383, 0.595], [0, 0]],
    'Cd': [[2.574, 55.675], [3.259, 11.838], [2.547, 2.784], [0.838, 0.322]],
    'Po': [[6.07, 28.075], [4.997, 4.999], [2.232, 0.563], [0, 0]],
    'Pm': [[5.201, 28.079], [4.094, 5.081], [1.719, 0.576], [0, 0]],
    'Ho': [[5.376, 28.773], [4.403, 5.174], [1.884, 0.582], [0, 0]],
    'Hf': [[5.588, 29.001], [4.619, 5.164], [1.997, 0.579], [0, 0]],
    'Hg': [[2.682, 42.822], [4.241, 9.856], [2.755, 2.295], [1.27, 0.307]],
    'He': [[0.091, 18.183], [0.181, 6.212], [0.11, 1.803], [0.036, 0.284]],
    'Mg': [[2.268, 73.67], [1.803, 20.175], [0.839, 3.013], [0.289, 0.405]],
    'K':  [[3.951, 137.075], [2.545, 22.402], [1.98, 4.532], [0.482, 0.434]],
    'Mn': [[2.747, 67.786], [2.456, 15.674], [1.792, 3.0], [0.498, 0.357]],
    'O':  [[0.455, 23.78], [0.917, 7.622], [0.472, 2.144], [0.138, 0.296]],
    'S':  [[1.659, 36.65], [2.386, 11.488], [0.79, 2.469], [0.321, 0.34]],
    'W':  [[5.709, 28.782], [4.677, 5.084], [2.019, 0.572], [0, 0]],
    'Zn': [[1.942, 54.162], [1.95, 12.518], [1.619, 2.416], [0.543, 0.33]],
    'Eu': [[6.267, 100.298], [4.844, 16.066], [3.202, 2.98], [1.2, 0.367]],
    'Zr': [[4.105, 28.492], [3.144, 5.277], [1.229, 0.601], [0, 0]],
    'Er': [[5.436, 28.655], [4.437, 5.117], [1.891, 0.577], [0, 0]],
    'Ni': [[2.21, 58.727], [2.134, 13.553], [1.689, 2.609], [0.524, 0.339]],
    'Na': [[2.241, 108.004], [1.333, 24.505], [0.907, 3.391], [0.286, 0.435]],
    'Nb': [[4.237, 27.415], [3.105, 5.074], [1.234, 0.593], [0, 0]],
    'Nd': [[5.151, 28.304], [4.075, 5.073], [1.683, 0.571], [0, 0]],
    'Ne': [[0.303, 17.64], [0.72, 5.86], [0.475, 1.762], [0.153, 0.266]],
    'Np': [[6.323, 29.142], [5.414, 5.096], [2.453, 0.568], [0, 0]],
    'Fr': [[6.201, 28.2], [5.121, 4.954], [2.275, 0.556], [0, 0]],
    'Fe': [[2.544, 64.424], [2.343, 14.88], [1.759, 2.854], [0.506, 0.35]],
    'B':  [[0.945, 46.444], [1.312, 14.178], [0.419, 3.223], [0.116, 0.377]],
    'F':  [[0.387, 20.239], [0.811, 6.609], [0.475, 1.931], [0.146, 0.279]],
    'Sr': [[5.848, 104.972], [4.003, 19.367], [2.342, 3.737], [0.88, 0.414]],
    'N':  [[0.572, 28.847], [1.043, 9.054], [0.465, 2.421], [0.131, 0.317]],
    'Kr': [[2.034, 29.999], [2.927, 9.598], [1.342, 1.952], [0.589, 0.299]],
    'Si': [[2.129, 57.775], [2.533, 16.476], [0.835, 2.88], [0.322, 0.386]],
    'Sn': [[3.45, 59.104], [3.735, 14.179], [2.118, 2.855], [0.877, 0.327]],
    'Sm': [[5.255, 28.016], [4.113, 5.037], [1.743, 0.577], [0, 0]],
    'V':  [[3.245, 76.379], [2.698, 17.726], [1.86, 3.363], [0.486, 0.374]],
    'Sc': [[3.966, 88.96], [2.917, 20.606], [1.925, 3.856], [0.48, 0.399]],
    'Sb': [[3.564, 50.487], [3.844, 13.316], [2.687, 2.691], [0.864, 0.316]],
    'Se': [[2.298, 38.83], [2.854, 11.536], [1.456, 2.146], [0.59, 0.316]],
    'Co': [[2.367, 61.431], [2.236, 14.18], [1.724, 2.725], [0.515, 0.344]],
    'Cm': [[6.46, 28.396], [5.469, 4.97], [2.471, 0.554], [0, 0]],
    'Cl': [[1.452, 30.935], [2.292, 9.98], [0.787, 2.234], [0.322, 0.323]],
    'Ca': [[4.47, 99.523], [2.971, 22.696], [1.97, 4.195], [0.482, 0.417]],
    'Cf': [[6.548, 28.461], [5.526, 4.965], [2.52, 0.557], [0, 0]],
    'Ce': [[5.007, 28.283], [3.98, 5.183], [1.678, 0.589], [0, 0]],
    'Xe': [[3.366, 35.509], [4.147, 11.117], [2.443, 2.294], [0.829, 0.289]],
    'Tm': [[5.441, 29.149], [4.51, 5.264], [1.956, 0.59], [0, 0]],
    'Cs': [[6.062, 155.837], [5.986, 19.695], [3.303, 3.335], [1.096, 0.379]],
    'Cr': [[2.307, 78.405], [2.334, 15.785], [1.823, 3.157], [0.49, 0.364]],
    'Cu': [[1.579, 62.94], [1.82, 12.453], [1.658, 2.504], [0.532, 0.333]],
    'La': [[4.94, 28.716], [3.968, 5.245], [1.663, 0.594], [0, 0]],
    'Li': [[1.611, 107.638], [1.246, 30.48], [0.326, 4.533], [0.099, 0.495]],
    'Tl': [[5.932, 29.086], [4.972, 5.126], [2.195, 0.572], [0, 0]],
    'Lu': [[5.553, 28.907], [4.58, 5.16], [1.969, 0.577], [0, 0]],
    'Th': [[6.264, 28.651], [5.263, 5.03], [2.367, 0.563], [0, 0]],
    'Ti': [[3.565, 81.982], [2.818, 19.049], [1.893, 3.59], [0.483, 0.386]],
    'Te': [[4.785, 27.999], [3.688, 5.083], [1.5, 0.581], [0, 0]],
    'Tb': [[5.272, 29.046], [4.347, 5.226], [1.844, 0.585], [0, 0]],
    'Tc': [[4.318, 28.246], [3.27, 5.148], [1.287, 0.59], [0, 0]],
    'Ta': [[5.659, 28.807], [4.63, 5.114], [2.014, 0.578], [0, 0]],
    'Yb': [[5.529, 28.927], [4.533, 5.144], [1.945, 0.578], [0, 0]],
    'Dy': [[5.332, 28.888], [4.37, 5.198], [1.863, 0.581], [0, 0]],
    'I':  [[3.473, 39.441], [4.06, 11.816], [2.522, 2.415], [0.84, 0.298]],
    'U':  [[6.767, 85.951], [6.729, 15.642], [4.014, 2.936], [1.561, 0.335]],
    'Y':  [[4.129, 27.548], [3.012, 5.088], [1.179, 0.591], [0, 0]],
    'Ac': [[6.278, 28.323], [5.195, 4.949], [2.321, 0.557], [0, 0]],
    'Ag': [[2.036, 61.497], [3.272, 11.824], [2.511, 2.846], [0.837, 0.327]],
    'Ir': [[5.754, 29.159], [4.851, 5.152], [2.096, 0.57], [0, 0]],
    'Am': [[6.378, 29.156], [5.495, 5.102], [2.495, 0.565], [0, 0]],
    'Al': [[2.276, 72.322], [2.428, 19.773], [0.858, 3.08], [0.317, 0.408]],
    'As': [[2.399, 45.718], [2.79, 12.817], [1.529, 2.28], [0.594, 0.328]],
    'Ar': [[1.274, 26.682], [2.19, 8.813], [0.793, 2.219], [0.326, 0.307]],
    'Au': [[2.388, 42.866], [4.226, 9.743], [2.689, 2.264], [1.255, 0.307]],
    'At': [[6.133, 28.047], [5.031, 4.957], [2.239, 0.558], [0, 0]],
    'In': [[3.153, 66.649], [3.557, 14.449], [2.818, 2.976], [0.884, 0.335]],
    'Mo': [[3.12, 72.464], [3.906, 14.642], [2.361, 3.237], [0.85, 0.366]]
}


class XRDCalculator(AbstractDiffractionPatternCalculator):
    '''
    Computes the XRD pattern of a crystal structure.

    This code is implemented by Shyue Ping Ong as part of UCSD's NANO106 -
    Crystallography of Materials. The formalism for this code is based on
    that given in Chapters 11 and 12 of Structure of Materials by Marc De
    Graef and Michael E. McHenry. This takes into account the atomic
    scattering factors and the Lorentz polarization factor, but not
    the Debye-Waller (temperature) factor (for which data is typically not
    available). Note that the multiplicity correction is not needed since
    this code simply goes through all reciprocal points within the limiting
    sphere, which includes all symmetrically equivalent facets. The algorithm
    is as follows

    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.

    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       \sin(\theta) = \frac{\lambda}{2d_{hkl}}

    3. Compute the structure factor as the sum of the atomic scattering
       factors. The atomic scattering factors are given by

        f(s) = Z - 41.78214 \times s^2 \times \sum\limits_{i=1}^n a_i \exp(-b_is^2)
        s = \frac{\sin(\theta)}{\lambda}
       
       The stucture factor is then given by
        F_{hkl} = \sum\limits_{j=1}^N f_j \exp(2\pi i\mathbf{g_{hkl}}\cdot \mathbf{r})

    4. The intensity is then given by the modulus square of the structure
       factor.
           I_{hkl} = F_{hkl}F_{hkl}^*


    '''

    # Tuple of available radiation keywords.
    AVAILABLE_RADIATION = tuple(WAVELENGTHS.keys())

    def __init__(self, wavelength=0.1, symprec=0, debye_waller_factors=None):
        '''
        Initializes the XRD calculator with a given radiation.

        Args:
            wavelength (str/float): The wavelength can be specified as either a
                float or a string. If it is a string, it must be one of the
                supported definitions in the AVAILABLE_RADIATION class
                variable, which provides useful commonly used wavelengths.
                If it is a float, it is interpreted as a wavelength in
                angstroms. Defaults to 'CuKa', i.e, Cu K_alpha radiation.
            symprec (float): Symmetry precision for structure refinement. If
                set to 0, no refinement is done. Otherwise, refinement is
                performed using spglib with provided precision.
            debye_waller_factors ({element symbol: float}): Allows the
                specification of Debye-Waller factors. Note that these
                factors are temperature dependent.
        '''
        if isinstance(wavelength, float):
            self.wavelength = wavelength
        else:
            self.radiation = wavelength
            self.wavelength = WAVELENGTHS[wavelength]
        self.symprec = symprec
        self.debye_waller_factors = debye_waller_factors or {}

    def get_pattern(self, structure, scaled=True, two_theta_range=None):
        '''
        Calculates the diffraction pattern for a structure.

        Args:
            structure (Structure): Input structure
            scaled (bool): Whether to return scaled intensities. The maximum
                peak is set to a value of 100. Defaults to True. Use False if
                you need the absolute values to combine XRD plots.
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.

        Returns:
            (XRDPattern)
        '''
        if self.symprec:
            finder = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = finder.get_refined_structure()

        wavelength = self.wavelength
        latt = structure.lattice
        is_hex = latt.is_hexagonal()

        # Create a flattened array of zs, coeffs, fcoords and occus. This is
        # used to perform vectorized computation of atomic scattering factors
        # later. Note that these are not necessarily the same size as the
        # structure as each partially occupied specie occupies its own
        # position in the flattened array.
        zs = []
        coeffs = []
        fcoords = []
        occus = []
        dwfactors = []

        for site in structure:
            for sp, occu in site.species.items():
                zs.append(sp.Z)
                try:
                    c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                except KeyError:
                    raise ValueError('Unable to calculate XRD pattern as '
                                     'there is no scattering coefficients for'
                                     ' %s.' % sp.symbol)
                coeffs.append(c)
                dwfactors.append(self.debye_waller_factors.get(sp.symbol, 0))
                fcoords.append(site.frac_coords)
                occus.append(occu)

        zs = np.array(zs)
        coeffs = np.array(coeffs)
        fcoords = np.array(fcoords)
        occus = np.array(occus)
        dwfactors = np.array(dwfactors)

        f_values = []
        hkls = []
        two_thetas = []

        for h in range(10):
            for k in range(10):
                for l in range(10):
                    hkl = np.array([h,k,l])
                    g_hkl =  1 / structure.lattice.d_hkl([h,k,l])

                    # Bragg condition
                    theta = asin(wavelength * g_hkl / 2)

                    # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 (d = 1/|ghkl|)
                    s = g_hkl / 2

                    # Store s^2 since we are using it a few times.
                    s2 = s ** 2

                    # Vectorized computation of g.r for all fractional coords and
                    # hkl.
                    g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]

                    # Highly vectorized computation of atomic scattering factors.
                    fs = zs - 41.78214 * s2 * np.sum(
                        coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1)

                    dw_correction = np.exp(-dwfactors * s2)

                    # Structure factor = sum of atomic scattering factors (with
                    # position factor exp(2j * pi * g.r and occupancies).
                    # Vectorized computation.
                    f_hkl = np.sum(fs * occus * np.exp(2j * pi * g_dot_r) * dw_correction)

                    # Intensity for hkl is modulus square of structure factor.
                    i_hkl = (f_hkl * f_hkl.conjugate()).real
                    two_theta = degrees(2 * theta)

                    f_values.append(i_hkl)
                    two_thetas.append(two_theta)
                    hkls.append(hkl)

        return f_values, hkls, two_thetas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data explore',
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', type=str, default='../MP_modulus.json',
                        help='the bulk modulus and structure from Materials Project')
    parser.add_argument('--feature_dir', type=str, default='./',
                        help='dir has all the input files')

    args = parser.parse_args()
    input_file = args.input_file
    #output_file = args.output_file
    feature_dir = args.feature_dir
    #task = args.task

    with open(input_file,'r') as f:
        data = json.load(f)

    x = XRDCalculator()

    for d in data:
        struct = Structure.from_str(d['cif'], fmt='cif')
        try:
            Fs, _, two_theta = x.get_pattern(struct)
            np.savetxt(d['task_id'], Fs, delimiter=' ', fmt='%.3f')
            #np.savetxt(d['task_id'] + '_hkl', hkl, delimiter=' ', fmt='%.3f')
            #np.savetxt(d['task_id'] + '_two_theta', two_theta, delimiter=' ', fmt='%.3f')
        except:
            print(d['task_id'], flush=True)

