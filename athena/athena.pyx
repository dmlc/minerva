cimport minerva as m
from cython.operator cimport dereference as deref
import sys
from libc.stdlib cimport calloc, free
from libc.string cimport strcpy
from libcpp.vector cimport vector

cdef vector[int] _list_to_scale(l):
    cdef vector[int] ret
    for i in l:
        ret.push_back(i)
    return ret

cdef NArray _wrap_cpp_narray(m.NArray n):
    ret = NArray()
    ret._d.assign(n)
    return ret

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
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            return _wrap_cpp_narray(
                m.narray_add_narray(deref(self._d), deref(r._d)))
        else:
            f = rhs
            return _wrap_cpp_narray(
                ret._d.assign(m.narray_add_num(deref(self._d), f)))

    def __radd__(NArray self, float lhs):
        return _wrap_cpp_narray(m.num_add_narray(lhs, deref(self._d)))

    def __iadd__(NArray self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.add_assign_narray(deref(r._d))
        else:
            f = rhs
            self._d.add_assign_num(f)

    def __sub__(NArray self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            return _wrap_cpp_narray(
                m.narray_sub_narray(deref(self._d), deref(r._d)))
        else:
            f = rhs
            return _wrap_cpp_narray(
                m.narray_sub_num(deref(self._d), f))

    def __rsub__(NArray self, float lhs):
        return _wrap_cpp_narray(m.num_sub_narray(lhs, deref(self._d)))

    def __isub__(NArray self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.sub_assign_narray(deref(r._d))
        else:
            f = rhs
            self._d.sub_assign_num(f)

    def __mul__(NArray self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            return _wrap_cpp_narray(
                m.narray_mul_narray(deref(self._d), deref(r._d)))
        else:
            f = rhs
            return _wrap_cpp_narray(
                m.narray_mul_num(deref(self._d), f))

    def __rmul__(NArray self, float lhs):
        return _wrap_cpp_narray(m.num_mul_narray(lhs, deref(self._d)))

    def __imul__(NArray self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.mul_assign_narray(deref(r._d))
        else:
            f = rhs
            self._d.mul_assign_num(f)

    def __div__(NArray self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            return _wrap_cpp_narray(
                m.narray_div_narray(deref(self._d), deref(r._d)))
        else:
            f = rhs
            return _wrap_cpp_narray(
                m.narray_div_num(deref(self._d), f))

    def __rdiv__(NArray self, float lhs):
        return _wrap_cpp_narray(m.num_div_narray(lhs, deref(self._d)))

    def __idiv__(NArray self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.div_assign_narray(deref(r._d))
        else:
            f = rhs
            self._d.div_assign_num(f)

    def sum(NArray self, rhs):
        cdef int i
        cdef vector[int] v
        # TODO yutian: use try catch to do type conversion
        if isinstance(rhs, int):
            i = rhs
            return _wrap_cpp_narray(self._d.sum_one(i))
        else:
            v = _list_to_scale(rhs)
            return _wrap_cpp_narray(self._d.sum_scale(m.ToScale(&v)))

    def max(NArray self, rhs):
        cdef int i
        cdef vector[int] v
        # TODO yutian: use try catch to do type conversion
        if isinstance(rhs, int):
            i = rhs
            return _wrap_cpp_narray(self._d.max_one(i))
        else:
            v = _list_to_scale(rhs)
            return _wrap_cpp_narray(self._d.max_scale(m.ToScale(&v)))

    def max_index(NArray self, int rhs):
        return _wrap_cpp_narray(self._d.max_index(rhs))

    property shape:
        def __get__(self):
            cdef vector[int] scale = m.OfScale(self._d.Size())
            return list(scale)

    @staticmethod
    def randn(s, float mean, float var):
        cdef vector[int] scale = _list_to_scale(s)
        return _wrap_cpp_narray(m.NArray.Randn(m.ToScale(&scale), mean, var))

cdef class PoolingAlgorithm:
    max = m.PoolingAlgorithmMax
    average = m.PoolingAlgorithmAverage

    @staticmethod
    cdef of_cpp(m.PoolingAlgorithm v):
        if m.PoolingAlgorithmEqual(v, m.PoolingAlgorithmMax):
            return PoolingAlgorithm.max
        else:
            return PoolingAlgorithm.average

cdef class PoolingInfo:
    cdef m.PoolingInfo* _d

    def __cinit__(
            self,
            PoolingAlgorithm a=PoolingAlgorithm.max,
            int h=0,
            int w=0,
            int sv=0,
            int sh=0,
            int ph=0,
            int pw=0):
        self._d = new m.PoolingInfo(a, h, w, sv, sh, ph, pw)

    def __dealloc__(self):
        del self._d

    property algorithm:
        def __set__(self, a):
            self._d.algorithm = a
        def __get__(self):
            cdef m.PoolingInfo* p
            p = self._d
            return PoolingAlgorithm.of_cpp(p.algorithm)

