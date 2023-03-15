=========
Changelog
=========

Version 0.3.0
=============

Major speed optimisations, particularly for computing EMD between GRID representations.

Backwards compatibility with previous versions is (almost) entirely maintained, but this version introduces
additional API methods. Future versions will likely use the newer array-based API approach, and deprecate
the older "list of dicts" approach.

Changes
-------

**EMD calculation**
- Numba-optimised calculation of EMD between GRID representations, both individually and computing
  entire EMD dissimilarity matrices. This makes particular use of cumulative GRID distributions internally.
  Additional features include memory-reduction techniques, for instance saving EMD results direct to disk
  (using e.g. h5py) without storing in memory.
      As an example, the calculation of a 12,000x12,000 EMD matrix based on 100 GRID shells takes approximately
      30 minutes on a desktop PC (Intel i5-8500, 6 cores) compared to ~30 days previously.
- Added initial ability to combine EMD between different GRID shells in different ways. Currently, a power-law 
  (shell_number**-n) is implemented to weight comparison equally between shells (n = 0) through to nearest neighbours
  dominating (n >= 1). A callable can also be provided to define the weights, but they must sum to n_shells.

**GRID generation**
- Major overhaul of GRID generation, so that Gaussian broadening and histogram binning are now handled using
  the same interface.
  This also fixes a bug where the KDE-broadened distances were not correctly integrated within
  each histogram bin, so narrow broadening oculd result in missing distribution. This is now improved to compute
  the KDE on a finer series of positions before later binning as desired, making it closer to a true integral.
- It is also now simpler to compute GRID representations without saving to disk, and collections of GRIDs can
  be saved/loaded as single files (e.g. npy or h5py).
- Initial implementation of a composition-based GRID and minor updates to composition handling.

**Other Changes**
- Improved testing, particularly of GRID generation and EMD computation
- Updated documentation
- Modified example scripts


Version 0.2.0
=============

- Changed to PyScaffold for development framework
- Added initial documentation, which is still in development
- A script to reproduce the results of the GRID publication is now
  available in the ``examples`` folder
- 


Version 0.1.3
=============

- Initial public release of gridrdf
- Functions modified to import as packages in the most part, and an initial bank of tests introduced
- Some optimisations to speed up computation and processing of GRIDs, for instance using multiprocessing
