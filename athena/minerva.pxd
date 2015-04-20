from libc.stdint cimport *
from libcpp cimport bool
from libcpp.vector cimport vector

#TODO yutian: numpy

cdef extern from './minerva_utils.h' namespace 'athena':
  uint64_t CreateCpuDevice() except +
  uint64_t CreateGpuDevice(int) except +
  int GetGpuDeviceCount() except +
  void WaitForAll() except +
  void SetDevice(uint64_t) except +
  Scale ToScale(vector[int]*) except +
  vector[int] OfScale(const Scale&) except +
  NArray FromNumpy(const float*, const Scale&) except +

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
  NArray NArrayAddNArray 'operator+'(const NArray&, const NArray&) except +
  NArray NArraySubNArray 'operator-'(const NArray&, const NArray&) except +
  NArray NArrayMulNArray 'operator*'(const NArray&, const NArray&) except +
  NArray NArrayDivNArray 'operator/'(const NArray&, const NArray&) except +
  NArray NumAddNArray 'operator+'(float, const NArray&) except +
  NArray NumSubNArray 'operator-'(float, const NArray&) except +
  NArray NumMulNArray 'operator*'(float, const NArray&) except +
  NArray NumDivNArray 'operator/'(float, const NArray&) except +
  NArray NArrayAddNum 'operator+'(const NArray&, float) except +
  NArray NArraySubNum 'operator-'(const NArray&, float) except +
  NArray NArrayMulNum 'operator*'(const NArray&, float) except +
  NArray NArrayDivNum 'operator/'(const NArray&, float) except +

  cppclass Scale:
    pass

  cppclass NArray:
    NArray() except +
    NArray assign 'operator='(const NArray&) except +
    NArray AddAssignNArray 'operator+='(const NArray&) except +
    NArray SubAssignNArray 'operator-='(const NArray&) except +
    NArray MulAssignNArray 'operator*='(const NArray&) except +
    NArray DivAssignNArray 'operator/='(const NArray&) except +
    NArray AddAssignNum 'operator+='(float) except +
    NArray SubAssignNum 'operator-='(float) except +
    NArray MulAssignNum 'operator*='(float) except +
    NArray DivAssignNum 'operator/='(float) except +
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

  ctypedef enum PoolingAlgorithm 'minerva::PoolingInfo::Algorithm':
    kPoolingAlgorithmMax 'minerva::PoolingInfo::Algorithm::kMax'
    kPoolingAlgorithmAverage 'minerva::PoolingInfo::Algorithm::kAverage'
  int OfPoolingAlgorithm 'athena::OfEvilEnumClass'(PoolingAlgorithm) except +
  PoolingAlgorithm ToPoolingAlgorithm\
    'athena::ToEvilEnumClass<minerva::PoolingInfo::Algorithm>'(int) except +

  ctypedef enum SoftmaxAlgorithm 'minerva::SoftmaxAlgorithm':
    kSoftmaxAlgorithmInstance 'minerva::SoftmaxAlgorithm::kInstance'
    kSoftmaxAlgorithmChannel 'minerva::SoftmaxAlgorithm::kChannel'
  int OfSoftmaxAlgorithm 'athena::OfEvilEnumClass'(SoftmaxAlgorithm) except +
  SoftmaxAlgorithm ToSoftmaxAlgorithm\
    'athena::ToEvilEnumClass<minerva::SoftmaxAlgorithm>'(int) except +

  ctypedef enum ActivationAlgorithm 'minerva::ActivationAlgorithm':
    kActivationAlgorithmSigmoid 'minerva::ActivationAlgorithm::kSigmoid'
    kActivationAlgorithmRelu 'minerva::ActivationAlgorithm::kRelu'
    kActivationAlgorithmTanh 'minerva::ActivationAlgorithm::kTanh'
  int OfActivationAlgorithm\
    'athena::OfEvilEnumClass'(ActivationAlgorithm) except +
  ActivationAlgorithm ToActivationAlgorithm\
    'athena::ToEvilEnumClass<minerva::ActivationAlgorithm>'(int) except +

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

