#include "narray.h"

using namespace std;

namespace minerva {

// lazy reductions
NArray NArray::Sum(int dim) {
  return Sum(Scale(dim));
}
NArray NArray::Sum(const Scale& dims) {
  // TODO
  return NArray();
}
NArray NArray::Max(int dim) {
  return Max(Scale(dim));
}
NArray NArray::Max(const Scale& dims) {
  // TODO
  return NArray();
}
NArray NArray::MaxIndex(int dim) {
  return MaxIndex(Scale(dim));
}
NArray NArray::MaxIndex(const Scale& dims) {
  // TODO
  return NArray();
}
// non-lazy reduction
float NArray::Sum() {
  // TODO
  return 0;
}
float NArray::Max() {
  // TODO
  return 0;
}
int NArray::CountZero() {
  // TODO
  return 0;
}

} // end of namespace minerva
