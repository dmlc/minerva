from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        '*',
        ['*.pyx'],
        language='c++',
        include_dirs=[
            '../minerva',
            '../deps'
        ],
        extra_compile_args=[
            '-std=c++11'
        ],
        libraries=[
            'minerva'
        ],
        library_dirs=[
            '../release/lib'
        ],
        runtime_library_dirs=[
            '../release/lib'
        ],
    )
]

setup(
    name='athena',
    ext_modules=cythonize(extensions)
)
