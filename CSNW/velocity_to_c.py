from setuptools import setup, Extension, dist
from Cython.Build import cythonize
import sys
import numpy

#dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])
setup(
    ext_modules = cythonize("velocity_matrix.pyx"),
    include_dirs=[numpy.get_include()]
)
setup(
    ext_modules = cythonize("velocity.pyx"),
    include_dirs=[numpy.get_include()]
)
