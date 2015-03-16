#!/usr/bin/env python
import os
from setuptools import setup, Extension

minerva_include_path = os.getcwd() + '/minerva'
minerva_lib_path = os.getcwd() + '/release/lib' # TODO: currently only use release version
ex_include_dirs = [minerva_include_path]
ex_lib_dirs = [minerva_lib_path]

config_file_path = os.path.join('configure.in')
with open(config_file_path) as config_file:
    for line in config_file.readlines():
        if line.find('INCLUDE') >= 0:
            in_p = line.split('=')[-1]
            # remove the " 
            in_p = in_p[1:len(in_p)-2]
            ex_include_dirs += [p.strip() for p in in_p.split(';')]
        elif line.find('LIB') >= 0:
            lib_p = line.split('=')[-1]
            lib_p = lib_p[1:len(lib_p)-2]
            ex_lib_dirs += [p.strip() for p in lib_p.split(';')]
        elif line.find('CUDA_ROOT') >= 0:
            cuda_in = line.split('=')[-1]
            ex_include_dirs.append(cuda_in.strip() + '/include')

setup(name='owl',
    version='1.0',
    maintainer='Minjie Wang',
    maintainer_email='minerva-support@googlegroups.com',
    license='Apache 2.0',
    url='https://github.com/minerva-developers/minerva',
    package_dir={'':'owl'},
    packages=['owl', 'owl.caffe'],
    ext_modules=[
        Extension('libowl',
            ['owl/owl.cpp'],
            language='c++',
            define_macros=[('HAS_CUDA', '1')], # TODO: must have cuda
            include_dirs=ex_include_dirs,
            libraries=['boost_python', 'boost_numpy', 'python2.7', 'minerva'],
            library_dirs=ex_lib_dirs,
            extra_compile_args=['-O2', '-std=c++11'],
            extra_link_args=[])
        ]
    )
