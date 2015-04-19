from libc.stdint cimport *
from libcpp.vector cimport vector

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
  cdef cppclass Scale:
    pass
  cdef cppclass NArray:
    NArray()
    @staticmethod
    NArray Randn(const Scale&, float, float) except +
    NArray assign 'operator='(NArray)
    NArray add_assign 'operator+='(NArray)
    NArray sub_assign 'operator-='(NArray)
    NArray mul_assign 'operator*='(NArray)
    NArray div_assign 'operator*='(NArray)
    NArray add_assign 'operator+='(float)
    NArray sub_assign 'operator-='(float)
    NArray mul_assign 'operator*='(float)
    NArray div_assign 'operator*='(float)
    Scale Size()

