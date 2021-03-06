# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import xgboost as xgb

setup(ext_modules = cythonize(Extension(
    'calculate_matrix',
    sources=['cal_mat.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))