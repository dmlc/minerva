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
        'libowl',
        sources=['owl/owl/libowl.pyx', 'owl/owl/minerva_utils.cpp'],
        language='c++',
        include_dirs=[
            'minerva',
            numpy.get_include(),
            'release/third_party/include',
        ],
        extra_compile_args=[
            '-std=c++11',
            '-Wall',
            '-O2',
            '-g'
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
    package_dir={'': 'owl/owl'},
    ext_modules=cythonize(extensions)
)
