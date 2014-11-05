#!/usr/bin/env python
import numpy as np
import libowl as _owl

initialize = _owl.initialize
create_cpu_device = _owl.create_cpu_device
create_gpu_device = _owl.create_gpu_device
set_device = _owl.set_device
wait_eval = _owl.wait_eval

zeros = _owl.zeros
ones = _owl.ones
randn = _owl.randn
make_narray = _owl.make_narray
def from_nparray(nparr):
    return _owl.make_narray([i for i in nparr.shape], nparr.T.flatten().tolist())

op = _owl.arithmetic

softmax = _owl.softmax
