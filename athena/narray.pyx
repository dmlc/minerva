cimport minerva
import sys

def initialize():
    cdef int argc = len(sys.argv)
    cdef char** argv = <char**>0
    minerva.Initialize(&argc, &argv)
