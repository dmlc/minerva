#!/usr/bin/env python
import numpy as np
import libowl as _owl

initialize = _owl.initialize
create_cpu_device = _owl.create_cpu_device
create_gpu_device = _owl.create_gpu_device
set_device = _owl.set_device

zeros = _owl.zeros
ones = _owl.ones
randn = _owl.randn
randb = _owl.randb
make_narray = _owl.make_narray

# Convert numpy array into minerva array. ATTENTION: this will lead to
# a transpose due to the different storage priority
#def from_nparray(nparr):
    #return _owl.make_narray(list(nparr.shape[::-1]), nparr.flatten().tolist())
from_nparray = _owl.from_nparray

op = _owl.arithmetic

softmax = _owl.softmax

