#!/usr/bin/env python

import libowl as _owl

SimpleFileLoader = _owl.SimpleFileLoader
FileFormat = _owl.FileFormat
MBLoader = _owl.MBLoader

logical_dag = _owl.logical_dag
initialize = _owl.initialize

load_from_file = _owl.load_from_file
zeros = _owl.zeros
ones = _owl.ones

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

softmax = _owl.softmax
