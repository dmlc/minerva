cimport minerva
import sys
from libc.stdlib cimport calloc, free
from libc.string cimport strcpy

def initialize():
    cdef int argc = len(sys.argv)
    cdef char** argv = <char**>(calloc(argc, sizeof(char*)))
    for i in range(argc):
        argv[i] = <char*>(calloc(len(sys.argv[i]) + 1, sizeof(char)))
        strcpy(argv[i], sys.argv[i])
    minerva.Initialize(&argc, &argv)
    for i in range(argc):
        free(argv[i])
    free(argv)
