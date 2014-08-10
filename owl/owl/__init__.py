#!/usr/bin/env python

import libowl as _owl

SimpleFileLoader = _owl.SimpleFileLoader
FileFormat = _owl.FileFormat

logical_dag = _owl.logical_dag
initialize = _owl.initialize

def load_from_file(size, path, loader, numparts):
    size = _owl.Scale(*size)
    numparts = _owl.Scale(*numparts)
    array = _owl.load_from_file(size, path, loader, numparts)
    return array
