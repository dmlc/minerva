cdef extern from './minerva_utils.h':
  void Test() except *

cdef extern from '../minerva/minerva.h' namespace 'minerva::MinervaSystem':
  void Initialize(int*, char***) except *
  void Finalize() except *

cdef extern from '../minerva/minerva.h' namespace 'minerva':
  NArray operator+(NArray, NArray)
  NArray operator-(NArray, NArray)
  NArray operator*(NArray, NArray)
  NArray operator/(NArray, NArray)
  NArray operator+(float, NArray)
  NArray operator-(float, NArray)
  NArray operator*(float, NArray)
  NArray operator/(float, NArray)
  NArray operator+(NArray, float)
  NArray operator-(NArray, float)
  NArray operator*(NArray, float)
  NArray operator/(NArray, float)
  cdef cppclass NArray:
    NArray()
    NArray add_assign "operator+="(NArray)
    NArray sub_assign "operator-="(NArray)
    NArray mul_assign "operator*="(NArray)
    NArray div_assign "operator*="(NArray)
    NArray add_assign "operator+="(float)
    NArray sub_assign "operator-="(float)
    NArray mul_assign "operator*="(float)
    NArray div_assign "operator*="(float)

