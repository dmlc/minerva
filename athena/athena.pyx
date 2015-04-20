cimport minerva as m
from cython.operator cimport dereference as deref
import sys
from libc.stdlib cimport calloc, free
from libc.string cimport strcpy
from libcpp.vector cimport vector

cdef vector[int] _list_to_vector(l):
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
                (m.narray_add_num(deref(self._d), f)))

    def __radd__(self, float lhs):
        return _wrap_cpp_narray(m.num_add_narray(lhs, deref(self._d)))

    def __iadd__(self, rhs):
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

    def __rsub__(self, float lhs):
        return _wrap_cpp_narray(m.num_sub_narray(lhs, deref(self._d)))

    def __isub__(self, rhs):
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

    def __rmul__(self, float lhs):
        return _wrap_cpp_narray(m.num_mul_narray(lhs, deref(self._d)))

    def __imul__(self, rhs):
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

    def __rdiv__(self, float lhs):
        return _wrap_cpp_narray(m.num_div_narray(lhs, deref(self._d)))

    def __idiv__(self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.div_assign_narray(deref(r._d))
        else:
            f = rhs
            self._d.div_assign_num(f)

    @staticmethod
    def mult(NArray lhs, NArray rhs):
        return _wrap_cpp_narray(m.Mult(deref(lhs._d), deref(rhs._d)))

    @staticmethod
    def exp(NArray lhs):
        return _wrap_cpp_narray(m.Exp(deref(lhs._d)))

    @staticmethod
    def ln(NArray lhs):
        return _wrap_cpp_narray(m.Ln(deref(lhs._d)))

    @staticmethod
    def sigm(NArray lhs):
        return _wrap_cpp_narray(m.SigmoidForward(deref(lhs._d)))

    @staticmethod
    def sigm_back(NArray diff, NArray top, NArray bottom):
        return _wrap_cpp_narray(
                m.SigmoidBackward(
                    deref(diff._d)
                ,   deref(top._d)
                ,   deref(bottom._d)))

    @staticmethod
    def relu(NArray lhs):
        return _wrap_cpp_narray(m.ReluForward(deref(lhs._d)))

    @staticmethod
    def relu_back(NArray diff, NArray top, NArray bottom):
        return _wrap_cpp_narray(
                m.ReluBackward(
                    deref(diff._d)
                ,   deref(top._d)
                ,   deref(bottom._d)))

    @staticmethod
    def tanh(NArray lhs):
        return _wrap_cpp_narray(m.TanhForward(deref(lhs._d)))

    @staticmethod
    def tanh_back(NArray diff, NArray top, NArray bottom):
        return _wrap_cpp_narray(
                m.TanhBackward(
                    deref(diff._d)
                ,   deref(top._d)
                ,   deref(bottom._d)))

    def sum(self, rhs):
        cdef int i
        cdef vector[int] v
        # TODO yutian: use try catch to do type conversion
        if isinstance(rhs, int):
            i = rhs
            return _wrap_cpp_narray(self._d.Sum(i))
        else:
            v = _list_to_vector(rhs)
            return _wrap_cpp_narray(self._d.Sum(m.ToScale(&v)))

    def max(self, rhs):
        cdef int i
        cdef vector[int] v
        # TODO yutian: use try catch to do type conversion
        if isinstance(rhs, int):
            i = rhs
            return _wrap_cpp_narray(self._d.Max(i))
        else:
            v = _list_to_vector(rhs)
            return _wrap_cpp_narray(self._d.Max(m.ToScale(&v)))

    def max_index(self, int rhs):
        return _wrap_cpp_narray(self._d.MaxIndex(rhs))

    def count_zero(self):
        return self._d.CountZero()

    def trans(self):
        return _wrap_cpp_narray(self._d.Trans())

    def reshape(self, s):
        cdef vector[int] v = _list_to_vector(s)
        return _wrap_cpp_narray(self._d.Reshape(m.ToScale(&v)))

    def wait_for_eval(self):
        self._d.Wait()

    property shape:
        def __get__(self):
            cdef vector[int] scale = m.OfScale(self._d.Size())
            return list(scale)

    @staticmethod
    def zeros(s):
        cdef vector[int] v = _list_to_vector(s)
        return _wrap_cpp_narray(m.NArray.Zeros(m.ToScale(&v)))

    @staticmethod
    def ones(s):
        cdef vector[int] v = _list_to_vector(s)
        return _wrap_cpp_narray(m.NArray.Ones(m.ToScale(&v)))

    @staticmethod
    def randn(s, float mean, float var):
        cdef vector[int] v = _list_to_vector(s)
        return _wrap_cpp_narray(m.NArray.Randn(m.ToScale(&v), mean, var))

    @staticmethod
    def randb(s, float p):
        cdef vector[int] v = _list_to_vector(s)
        return _wrap_cpp_narray(m.NArray.RandBernoulli(m.ToScale(&v), p))

cdef class _PoolingAlgorithm(object):
    cdef m.PoolingAlgorithm max
    cdef m.PoolingAlgorithm average
    def __cinit__(self):
        self.max = m.kPoolingAlgorithmMax
        self.average = m.kPoolingAlgorithmAverage

PoolingAlgorithm = _PoolingAlgorithm()
    # max = m.PoolingAlgorithmMax
    # average = m.PoolingAlgorithmAverage

    # @staticmethod
    # cdef of_cpp(m.PoolingAlgorithm v):
    #     if m.PoolingAlgorithmEqual(v, m.PoolingAlgorithmMax):
    #         return PoolingAlgorithm.max
    #     else:
    #         return PoolingAlgorithm.average

# cdef class SoftmaxAlgorithm(object):
#     instance = m.SoftmaxAlgorithmInstance
#     channel = m.SoftmaxAlgorithmChannel

# cdef class PoolingAlgorithm:
#     max = m.PoolingAlgorithmMax
#     average = m.PoolingAlgorithmAverage

# cdef class PoolingInfo(object):
#     cdef m.PoolingInfo* _d
#
#     def __cinit__(
#             self,
#             PoolingAlgorithm a=PoolingAlgorithm.max,
#             int h=0,
#             int w=0,
#             int sv=0,
#             int sh=0,
#             int ph=0,
#             int pw=0):
#         self._d = new m.PoolingInfo(a, h, w, sv, sh, ph, pw)
#
#     def __dealloc__(self):
#         del self._d
#
#     property algorithm:
#         def __set__(self, a):
#             self._d.algorithm = a
#         def __get__(self):
#             cdef m.PoolingInfo* p
#             p = self._d
#             return PoolingAlgorithm.of_cpp(p.algorithm)

