from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp cimport bool

#TODO yutian: numpy

cdef extern from './minerva_utils.h' namespace 'athena':
  uint64_t CreateCpuDevice() except +
  uint64_t CreateGpuDevice(int) except +
  int GetGpuDeviceCount() except +
  void WaitForAll() except +
  void SetDevice(uint64_t) except +
  Scale ToScale(vector[int]*) except +
  vector[int] OfScale(const Scale&) except +

cdef extern from '../minerva/minerva.h' namespace 'minerva::MinervaSystem':
  void Initialize(int*, char***) except +
  void Finalize() except +

cdef extern from '../minerva/minerva.h' namespace 'minerva::Elewise':
  NArray Mult(const NArray&, const NArray&) except +
  NArray Exp(const NArray&) except +
  NArray Ln(const NArray&) except +
  NArray SigmoidForward(const NArray&) except +
  NArray SigmoidBackward(const NArray&, const NArray&, const NArray&) except +
  NArray ReluForward(const NArray&) except +
  NArray ReluBackward(const NArray&, const NArray&, const NArray&) except +
  NArray TanhForward(const NArray&) except +
  NArray TanhBackward(const NArray&, const NArray&, const NArray&) except +

cdef extern from '../minerva/minerva.h' namespace 'minerva':
  NArray narray_add_narray 'operator+'(const NArray&, const NArray&) except +
  NArray narray_sub_narray 'operator-'(const NArray&, const NArray&) except +
  NArray narray_mul_narray 'operator*'(const NArray&, const NArray&) except +
  NArray narray_div_narray 'operator/'(const NArray&, const NArray&) except +
  NArray num_add_narray 'operator+'(float, const NArray&) except +
  NArray num_sub_narray 'operator-'(float, const NArray&) except +
  NArray num_mul_narray 'operator*'(float, const NArray&) except +
  NArray num_div_narray 'operator/'(float, const NArray&) except +
  NArray narray_add_num 'operator+'(const NArray&, float) except +
  NArray narray_sub_num 'operator-'(const NArray&, float) except +
  NArray narray_mul_num 'operator*'(const NArray&, float) except +
  NArray narray_div_num 'operator/'(const NArray&, float) except +

  cppclass Scale:
    pass

  cppclass NArray:
    NArray() except +
    NArray assign 'operator='(const NArray&) except +
    NArray add_assign_narray 'operator+='(const NArray&) except +
    NArray sub_assign_narray 'operator-='(const NArray&) except +
    NArray mul_assign_narray 'operator*='(const NArray&) except +
    NArray div_assign_narray 'operator/='(const NArray&) except +
    NArray add_assign_num 'operator+='(float) except +
    NArray sub_assign_num 'operator-='(float) except +
    NArray mul_assign_num 'operator*='(float) except +
    NArray div_assign_num 'operator/='(float) except +
    NArray Sum(int) except +
    NArray Sum(const Scale&) except +
    NArray Max(int) except +
    NArray Max(const Scale&) except +
    NArray MaxIndex(int) except +
    int CountZero() except +
    NArray Trans() except +
    NArray Reshape(const Scale&) except +
    void Wait() except +
    Scale Size() except +
    @staticmethod
    NArray Zeros(const Scale&) except +
    @staticmethod
    NArray Ones(const Scale&) except +
    @staticmethod
    NArray Randn(const Scale&, float, float) except +
    @staticmethod
    NArray RandBernoulli(const Scale&, float) except +

  # TODO yutian: RI LE GOU, see if there is a better way
  ctypedef enum PoolingAlgorithm 'minerva::PoolingInfo::Algorithm':
    kPoolingAlgorithmMax 'minerva::PoolingInfo::Algorithm::kMax'
    kPoolingAlgorithmAverage 'minerva::PoolingInfo::Algorithm::kAverage'

  ctypedef enum SoftmaxAlgorithm 'minerva::SoftmaxAlgorithm':
    kSoftmaxAlgorithmInstance 'SoftmaxAlgorithm::kInstance'
    kSoftmaxAlgorithmChannel 'SoftmaxAlgorithm::kChannel'

  ctypedef enum ActivationAlgorithm 'minerva::ActivationAlgorithm':
    kActivationAlgorithmSigmoid 'ActivationAlgorithm::kSigmoid'
    kActivationAlgorithmRelu 'ActivationAlgorithm::kRelu'
    kActivationAlgorithmTanh 'ActivationAlgorithm::kTanh'

  bool PoolingAlgorithmEqual 'athena::EnumClassEqual'(PoolingAlgorithm
  , PoolingAlgorithm)
  bool SoftmaxAlgorithm 'athena::EnumClassEqual'(SoftmaxAlgorithm
  , SoftmaxAlgorithm)
  bool ActivationAlgorithmEqual 'athena::EnumClassEqual'(ActivationAlgorithm
  , ActivationAlgorithm)

  cppclass ConvInfo:
    ConvInfo(int, int, int, int)
    int pad_height
    int pad_width
    int stride_vertical
    int stride_horizontal

  cppclass PoolingInfo:
    PoolingInfo(PoolingAlgorithm, int, int, int, int, int, int)
    PoolingAlgorithm algorithm
    int height
    int width
    int stride_vertical
    int stride_horizontal
    int pad_height
    int pad_width

