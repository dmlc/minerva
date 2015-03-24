#!/usr/bin/env python
import os
import shutil
import logging
from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext

logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger()

def copy_lib(build_dir, lib_name):
    """ copy libminerva.so and libowl.so into source dir
    
    This command is prior to running a bdist
    """
    root = os.path.dirname(__file__)
    local = os.path.abspath(os.path.join(root, 'owl', 'owl'))
    try:
        lib = os.path.realpath(os.path.join(build_dir, 'lib', lib_name))
        print ("copying %s -> %s" % (lib, local))
        shutil.copy(lib, local)
    except Exception:
        if not os.path.exists(local):
            logger.error("Fatal: local path not exist.")
        if not os.path.exists(lib):
            msg = ("%s not found. Please build c++ library first." % lib)
            logger.error("Fatal: " + msg)

package_data = {'owl': ["libowl.so"]}

setup(name='owl',
    version='1.0',
    maintainer='Minjie Wang',
    maintainer_email='minerva-support@googlegroups.com',
    license='Apache 2.0',
    url='https://github.com/minerva-developers/minerva',
    package_dir={'':'owl'},
    packages=['owl', 'owl.caffe'],
    package_data=package_data,
    zip_safe=False
    )
