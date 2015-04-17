from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        '*',
        ['*.pyx'],
        include_dirs=[
            '../minerva',
            '../deps'
        ],
        extra_compile_args=[
            '-std=c++11'
        ],
        libraries=[],
        library_dirs=[]
    )
]

setup(
    name='athena',
    ext_modules=cythonize(extensions)
)
