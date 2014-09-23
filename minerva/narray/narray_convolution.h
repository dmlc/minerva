#pragma once
#include "narray/narray.h"

namespace minerva {

class Convolution {
 public:
  static NArray ConvFF(NArray, NArray, const ConvInfo&);
  static NArray ConvBP(NArray, NArray, const ConvInfo&);
  static NArray GetGrad(NArray, NArray, const ConvInfo&);
};

}

