import sys
from cython.operator cimport dereference as deref
import cython
from libc.stdlib cimport calloc, free
from libc.string cimport strcpy
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
cimport minerva as m

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

def has_cuda():
    return m.has_cuda

cdef class NArray(object):
    cdef m.NArray* _d

    def __cinit__(self):
        self._d = new m.NArray()

    def __dealloc__(self):
        del self._d

    def __add__(self, rhs):
        cdef NArray l
        cdef NArray r
        cdef float f
        if isinstance(self, NArray):
            l = self
            if isinstance(rhs, NArray):
                r = rhs
                return _wrap_cpp_narray(
                    m.NArrayAddNArray(deref(l._d), deref(r._d)))
            else:
                f = rhs
                return _wrap_cpp_narray(
                    (m.NArrayAddNum(deref(l._d), f)))
        else:
            f = self
            r = rhs
            return _wrap_cpp_narray(m.NumAddNArray(f, deref(r._d)))

    def __iadd__(self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.AddAssignNArray(deref(r._d))
        else:
            f = rhs
            self._d.AddAssignNum(f)
        return self

    def __sub__(self, rhs):
        cdef NArray l
        cdef NArray r
        cdef float f
        if isinstance(self, NArray):
            l = self
            if isinstance(rhs, NArray):
                r = rhs
                return _wrap_cpp_narray(
                    m.NArraySubNArray(deref(l._d), deref(r._d)))
            else:
                f = rhs
                return _wrap_cpp_narray(
                    m.NArraySubNum(deref(l._d), f))
        else:
            f = self
            r = rhs
            return _wrap_cpp_narray(m.NumSubNArray(f, deref(r._d)))

    def __isub__(self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.SubAssignNArray(deref(r._d))
        else:
            f = rhs
            self._d.SubAssignNum(f)
        return self

    def __mul__(self, rhs):
        cdef NArray l
        cdef NArray r
        cdef float f
        if isinstance(self, NArray):
            l = self
            if isinstance(rhs, NArray):
                r = rhs
                return _wrap_cpp_narray(
                    m.NArrayMulNArray(deref(l._d), deref(r._d)))
            else:
                f = rhs
                return _wrap_cpp_narray(
                    m.NArrayMulNum(deref(l._d), f))
        else:
            f = self
            r = rhs
            return _wrap_cpp_narray(m.NumMulNArray(f, deref(r._d)))

    def __imul__(self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.MulAssignNArray(deref(r._d))
        else:
            f = rhs
            self._d.MulAssignNum(f)
        return self

    def __div__(self, rhs):
        cdef NArray l
        cdef NArray r
        cdef float f
        if isinstance(self, NArray):
            l = self
            if isinstance(rhs, NArray):
                r = rhs
                return _wrap_cpp_narray(
                    m.NArrayDivNArray(deref(l._d), deref(r._d)))
            else:
                f = rhs
                return _wrap_cpp_narray(
                    m.NArrayDivNum(deref(l._d), f))
        else:
            f = self
            r = rhs
            return _wrap_cpp_narray(m.NumDivNArray(f, deref(r._d)))

    def __idiv__(self, rhs):
        cdef NArray r
        cdef float f
        if isinstance(rhs, NArray):
            r = rhs
            self._d.DivAssignNArray(deref(r._d))
        else:
            f = rhs
            self._d.DivAssignNum(f)
        return self

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

    @staticmethod
    def conv_forward(NArray src, NArray filter, NArray bias, ConvInfo info):
        return _wrap_cpp_narray(
                m.ConvForward(
                    deref(src._d)
                ,   deref(filter._d)
                ,   deref(bias._d)
                ,   deref(info._d)));

    @staticmethod
    def conv_backward_data(
            NArray diff, NArray bottom, NArray filter, ConvInfo info):
        return _wrap_cpp_narray(
                m.ConvBackwardData(
                    deref(diff._d)
                ,   deref(bottom._d)
                ,   deref(filter._d)
                ,   deref(info._d)));

    @staticmethod
    def conv_backward_filter(
            NArray diff, NArray bottom, NArray filter, ConvInfo info):
        return _wrap_cpp_narray(
                m.ConvBackwardFilter(
                    deref(diff._d)
                ,   deref(bottom._d)
                ,   deref(filter._d)
                ,   deref(info._d)));

    @staticmethod
    def conv_backward_bias(NArray diff):
        return _wrap_cpp_narray(m.ConvBackwardBias(deref(diff._d)));

    @staticmethod
    def softmax_forward(NArray src, SoftmaxAlgorithmWrapper algo):
        return _wrap_cpp_narray(
                m.SoftmaxForward(
                    deref(src._d)
                ,   m.ToSoftmaxAlgorithm(algo._d)));

    @staticmethod
    def softmax_backward(NArray diff, NArray top, SoftmaxAlgorithmWrapper algo):
        return _wrap_cpp_narray(
                m.SoftmaxBackward(
                    deref(diff._d)
                ,   deref(top._d)
                ,   m.ToSoftmaxAlgorithm(algo._d)));

    @staticmethod
    def activation_forward(NArray src, ActivationAlgorithmWrapper algo):
        return _wrap_cpp_narray(
                m.ActivationForward(
                    deref(src._d)
                ,   m.ToActivationAlgorithm(algo._d)));

    @staticmethod
    def activation_backward(
            NArray diff
        ,   NArray top
        ,   NArray bottom
        ,   ActivationAlgorithmWrapper algo):
        return _wrap_cpp_narray(
                m.ActivationBackward(
                    deref(diff._d)
                ,   deref(top._d)
                ,   deref(bottom._d)
                ,   m.ToActivationAlgorithm(algo._d)));

    @staticmethod
    def pooling_forward(NArray src, PoolingInfo algo):
        return _wrap_cpp_narray(
                m.PoolingForward(
                    deref(src._d)
                ,   deref(algo._d)));

    @staticmethod
    def pooling_backward(
            NArray diff
        ,   NArray top
        ,   NArray bottom
        ,   PoolingInfo algo):
        return _wrap_cpp_narray(
                m.PoolingBackward(
                    deref(diff._d)
                ,   deref(top._d)
                ,   deref(bottom._d)
                ,   deref(algo._d)));

    @staticmethod
    def lrn_forward(NArray src, NArray scale, int local_size, float a, float b):
        return _wrap_cpp_narray(
                m.LRNForward(
                    deref(src._d)
                ,   deref(scale._d)
                ,   local_size
                ,   a
                ,   b));

    @staticmethod
    def lrn_backward(
            NArray bottom_data
        ,   NArray top_data
        ,   NArray scale
        ,   NArray top_diff
        ,   int local_size
        ,   float a
        ,   float b):
        return _wrap_cpp_narray(
                m.LRNBackward(
                    deref(bottom_data._d)
                ,   deref(top_data._d)
                ,   deref(scale._d)
                ,   deref(top_diff._d)
                ,   local_size
                ,   a
                ,   b));

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

    @staticmethod
    def concat(arrays, int dim):
        cdef vector[m.NArray] v
        cdef NArray n
        for i in arrays:
            n = i
            v.push_back(deref(n._d))
        return _wrap_cpp_narray(m.Concat(v, dim))

    @staticmethod
    def slice(NArray n, int slice_dim, int st_off, int slice_count):
        return _wrap_cpp_narray(
                m.Slice(deref(n._d), slice_dim, st_off, slice_count))

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def from_numpy(np.ndarray n):
        s = list(np.shape(n))
        n = n.astype(np.float32)
        return NArray._from_numpy(n.flatten(), s)

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _from_numpy(np.ndarray[np.float32_t, ndim=1, mode='c'] n, list s):
        cdef vector[int] shape = _list_to_vector(reversed(s))
        return _wrap_cpp_narray(m.FromNumpy(&n[0], m.ToScale(&shape)))

    def to_numpy(self):
        cdef int size = 1
        for i in self.shape:
            size *= i
        cdef np.ndarray[np.float32_t, ndim=1, mode='c'] dest
        dest = np.empty(size, dtype=np.float32, order='c')
        m.ToNumpy(&dest[0], deref(self._d))
        return dest.reshape(tuple(reversed(self.shape)))

cdef class PoolingAlgorithmWrapper(object):
    cdef int _d

    def __cinit__(self, int d):
        self._d = d

    def is_same(self, PoolingAlgorithmWrapper rhs):
        return self._d == rhs._d

class _PoolingAlgorithms(object):
    def __init__(self):
        self._max = PoolingAlgorithmWrapper(
                m.OfPoolingAlgorithm(m.kPoolingAlgorithmMax))
        self._average = PoolingAlgorithmWrapper(
                m.OfPoolingAlgorithm(m.kPoolingAlgorithmAverage))

    def find(self, a):
        if self._max.is_same(a):
            return self._max
        elif self._average.is_same(a):
            return self._average
        else:
            raise TypeError('invalid pooling algorithm')

    @property
    def max(self):
        return self._max

    @property
    def average(self):
        return self._average

pooling_algo = _PoolingAlgorithms()

cdef class SoftmaxAlgorithmWrapper(object):
    cdef int _d

    def __cinit__(self, int d):
        self._d = d

    def is_same(self, SoftmaxAlgorithmWrapper rhs):
        return self._d == rhs._d

class _SoftmaxAlgorithms(object):
    def __init__(self):
        self._instance = SoftmaxAlgorithmWrapper(
                m.OfSoftmaxAlgorithm(m.kSoftmaxAlgorithmInstance))
        self._channel = SoftmaxAlgorithmWrapper(
                m.OfSoftmaxAlgorithm(m.kSoftmaxAlgorithmChannel))

    def find(self, a):
        if self._instance.is_same(a):
            return self._instance
        elif self._channel.is_same(a):
            return self._channel
        else:
            raise TypeError('invalid softmax algorithm')

    @property
    def instance(self):
        return self._instance

    @property
    def channel(self):
        return self._channel

softmax_algo = _SoftmaxAlgorithms()

cdef class ActivationAlgorithmWrapper(object):
    cdef int _d

    def __cinit__(self, int d):
        self._d = d

    def is_same(self, ActivationAlgorithmWrapper rhs):
        return self._d == rhs._d

class _ActivationAlgorithms(object):
    def __init__(self):
        self._sigmoid = ActivationAlgorithmWrapper(
                m.OfActivationAlgorithm(m.kActivationAlgorithmSigmoid))
        self._relu = ActivationAlgorithmWrapper(
                m.OfActivationAlgorithm(m.kActivationAlgorithmRelu))
        self._tanh = ActivationAlgorithmWrapper(
                m.OfActivationAlgorithm(m.kActivationAlgorithmTanh))

    def find(self, a):
        if self._sigmoid.is_same(a):
            return self._sigmoid
        elif self._relu.is_same(a):
            return self._relu
        elif self._relu.is_same(a):
            return self._tanh
        else:
            raise TypeError('invalid activation algorithm')

    @property
    def sigmoid(self):
        return self._sigmoid

    @property
    def relu(self):
        return self._relu

    @property
    def tanh(self):
        return self._tanh

activation_algo = _ActivationAlgorithms()

cdef class ConvInfo(object):
    cdef m.ConvInfo* _d

    def __cinit__(
            self
        ,   int ph=0
        ,   int pw=0
        ,   int sv=1
        ,   int sh=1):
        self._d = new m.ConvInfo(ph, pw, sv, sh)

    def __dealloc__(self):
        del self._d

    property pad_height:
        def __set__(self, ph):
            self._d.pad_height = ph

        def __get__(self):
            return self._d.pad_height

    property pad_width:
        def __set__(self, pw):
            self._d.pad_width = pw

        def __get__(self):
            return self._d.pad_width

    property stride_vertical:
        def __set__(self, sv):
            self._d.stride_vertical = sv

        def __get__(self):
            return self._d.stride_vertical

    property stride_horizontal:
        def __set__(self, sh):
            self._d.stride_horizontal = sh

        def __get__(self):
            return self._d.stride_horizontal

cdef class PoolingInfo(object):
    cdef m.PoolingInfo* _d

    def __cinit__(
            self
        ,   PoolingAlgorithmWrapper a=pooling_algo.max
        ,   int h=0
        ,   int w=0
        ,   int sv=0
        ,   int sh=0
        ,   int ph=0
        ,   int pw=0):
        cdef m.PoolingAlgorithm algo
        algo = m.ToPoolingAlgorithm(a._d)
        self._d = new m.PoolingInfo(algo, h, w, sv, sh, ph, pw)

    def __dealloc__(self):
        del self._d

    property algorithm:
        def __set__(self, PoolingAlgorithmWrapper a):
            cdef m.PoolingAlgorithm algo
            algo = m.ToPoolingAlgorithm(a._d)
            self._d.algorithm = algo

        def __get__(self):
            return pooling_algo.find(PoolingAlgorithmWrapper(
                    m.OfPoolingAlgorithm(self._d.algorithm)))

    property height:
        def __set__(self, h):
            self._d.height = h

        def __get__(self):
            return self._d.height

    property width:
        def __set__(self, w):
            self._d.width = w

        def __get__(self):
            return self._d.width

    property stride_vertical:
        def __set__(self, sv):
            self._d.stride_vertical = sv

        def __get__(self):
            return self._d.stride_vertical

    property stride_horizontal:
        def __set__(self, sh):
            self._d.stride_horizontal = sh

        def __get__(self):
            return self._d.stride_horizontal

    property pad_height:
        def __set__(self, ph):
            self._d.pad_height = ph

        def __get__(self):
            return self._d.pad_height

    property pad_width:
        def __set__(self, pw):
            self._d.pad_width = pw

        def __get__(self):
            return self._d.pad_width

