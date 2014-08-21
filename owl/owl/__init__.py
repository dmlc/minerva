#!/usr/bin/env python

import libowl as _owl

SimpleFileLoader = _owl.SimpleFileLoader
FileFormat = _owl.FileFormat
MBLoader = _owl.MBLoader

logical_dag = _owl.logical_dag
initialize = _owl.initialize
wait_eval = _owl.wait_eval

load_from_file = _owl.load_from_file
zeros = _owl.zeros
ones = _owl.ones
make_narray = _owl.make_narray
to_list = _owl.to_list

op = _owl.arithmetic

def zeros(shape):
    num_parts = [1 for i in shape]
    return _owl.zeros(shape, num_parts)

def ones(shape):
    num_parts = [1 for i in shape]
    return _owl.ones(shape, num_parts)

def load_from_file(shape, fname, loader):
    num_parts = [1 for i in shape]
    return _owl.load_from_file(shape, fname, loader, num_parts)

def make_narray(shape, val):
    num_parts = [1 for i in shape]
    return _owl.make_narray(shape, val, num_parts)

softmax = _owl.softmax
