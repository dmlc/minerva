from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        '*',
        ['*.pyx'],
        include_dirs=['../minerva']
    )
]

setup(
    name='athena',
    ext_modules=cythonize(extensions)
)
