cimport minerva as m
from cython.operator cimport dereference as deref
import sys
from libc.stdlib cimport calloc, free
from libc.string cimport strcpy
from libcpp.vector cimport vector

def create_cpu_device():
    return m.CreateCpuDevice()

def create_gpu_device(i):
    return m.CreateGpuDevice(i)

def get_gpu_device_count():
    return m.GetGpuDeviceCount()

def wait_for_all():
    m.WaitForAll()

def set_device(i):
    m.SetDevice(i)

def initialize():
    cdef int argc = len(sys.argv)
    cdef char** argv = <char**>(calloc(argc, sizeof(char*)))
    for i in range(argc):
        argv[i] = <char*>(calloc(len(sys.argv[i]) + 1, sizeof(char)))
        strcpy(argv[i], sys.argv[i])
    m.Initialize(&argc, &argv)
    for i in range(argc):
        free(argv[i])
    free(argv)

def finalize():
    m.Finalize()

cdef class NArray(object):
    cdef m.NArray* _d
    def __cinit__(self):
        self._d = new m.NArray()
    def __dealloc__(self):
        del self._d
    def __add__(NArray self, rhs):
        cdef NArray r
        cdef m.NArray* d = self._d
        cdef float f
        ret = NArray()
        if isinstance(rhs, NArray):
            r = rhs
            ret._d.assign(m.narray_add_narray(deref(d), deref(r._d)))
        else:
            f = rhs
            ret._d.assign(m.narray_add_num(deref(d), f))
        return ret
    def __radd__(NArray self, float lhs):
        cdef m.NArray* d = self._d
        ret = NArray()
        ret._d.assign(m.num_add_narray(lhs, deref(d)))
        return ret
    @staticmethod
    def randn(s, float mean, float var):
        cdef vector[int] scale
        for i in s:
            scale.push_back(i)
        ret = NArray()
        ret._d.assign(m.NArray.Randn(m.ToScale(&scale), mean, var))
        return ret
    property shape:
        def __get__(self):
            cdef vector[int] scale = m.OfScale(self._d.Size())
            return list(scale)

