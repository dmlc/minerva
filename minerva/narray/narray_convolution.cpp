#include "narray/narray_convolution.h"
#include <glog/logging.h>

using namespace std;

namespace minerva {

NArray Convolution::ConvFF(NArray, NArray, const ConvInfo&) {
  // TODO
  CHECK(false) << "not implemented";
  return NArray();
}

NArray Convolution::ConvBP(NArray, NArray, const ConvInfo&) {
  // TODO
  CHECK(false) << "not implemented";
  return NArray();
}

NArray Convolution::GetGrad(NArray, NArray, const ConvInfo&) {
  // TODO
  CHECK(false) << "not implemented";
  return NArray();
}

}

