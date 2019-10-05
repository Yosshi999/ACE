import os
from distutils.core import setup

import numpy as np
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(os.path.join('ace', 'slic', '_slic.pyx')),
    include_dirs=[np.get_include()],
)
