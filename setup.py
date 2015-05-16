#!/usr/bin/env python
from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import numpy

# Hack to use specified compiler
os.environ['CC'] = 'g++'
os.environ['OPT'] = ''

def relative_path(to):
    base = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base, to)

extensions = [
    Extension(
        '*',
        ['owl/*.pyx', 'owl/minerva_utils.cpp'],
        language='c++',
        include_dirs=[
            'minerva',
            numpy.get_include()
        ],
        extra_compile_args=[
            '-std=c++11',
            '-Wall',
            '-O2',
            '-g'
        ],
        define_macros=[
            ('HAS_CUDA', None),
        ],
        libraries=[
            'minerva'
        ],
        library_dirs=[
            'release/lib'
        ],
        runtime_library_dirs=map(relative_path, [
            'release/lib'
        ]),
    )
]

setup(
    name='owl',
    package_dir={'': 'owl'},
    ext_modules=cythonize(extensions)
)
