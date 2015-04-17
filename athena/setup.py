from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        '*',
        ['*.pyx', 'minerva_utils.cpp'],
        language='c++',
        include_dirs=[
            '../minerva',
            '../deps',
            '/usr/local/cuda/include',
            '/home/yutian/cpp/cudnn-6.5-linux-x64-v2'
        ],
        extra_compile_args=[
            '-std=c++11'
        ],
        define_macros=[
            ('HAS_CUDA', None)
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
