""" Grouped Representations of Interatomic Distances (GRID)

Package to compute, analyse and train ML models using the GRID descriptor
for crystal structures.


"""

__version__ = "0.1.3"


__all__ = ['composition',
           'data_explore',
           'data_io',
           'data_prepare',
           'earth_mover_distance',
           'extendRDF',
           'otherRDFs',
           'train',
           'visualization',
           ]

from . import composition
from . import data_explore
from . import data_io
from . import data_prepare
from . import earth_mover_distance
from . import extendRDF
from . import otherRDFs
from . import train
from . import visualization