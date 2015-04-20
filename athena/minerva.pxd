from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp cimport bool

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
    NArray sum_one 'Sum'(int) except +
    NArray sum_scale 'Sum'(const Scale&) except +
    NArray max_one 'Max'(int) except +
    NArray max_scale 'Max'(const Scale&) except +
    NArray max_index 'MaxIndex'(int) except +
    Scale Size() except +
    @staticmethod
    NArray Randn(const Scale&, float, float) except +
  # TODO RI LE GOU
  ctypedef enum PoolingAlgorithm 'minerva::PoolingInfo::Algorithm':
    kPoolingAlgorithmMax 'PoolingInfo::Algorithm::kMax'
    kPoolingAlgorithmAverage 'PoolingInfo::Algorithm::kAverage'
  bool PoolingAlgorithmEqual 'athena::EnumClassEqual'(PoolingAlgorithm, PoolingAlgorithm)
  cppclass PoolingInfo:
    PoolingInfo(PoolingAlgorithm, int, int, int, int, int, int)
    PoolingAlgorithm algorithm
    int height
    int width
    int stride_vertical
    int stride_horizontal
    int pad_height
    int pad_width

