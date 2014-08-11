#pragma once

#include "common/scale.h"
#include "impl/impl.h"

namespace minerva {

struct Place {
  int procid;
  int device_type; // 0 is CPU, 1 is GPU
  int device_id; // which core or which GPU
};

const Place kUnknownPlace = {-1, -1, -1};

struct OpNodeContext {
  Place place;
  ImplType impl_type; // -1 is dynamic, 0 is basic, 1 is MKL, 2 is CUDA
};

struct DataNodeContext {
  Place place;
  //bool transpose;
};

inline bool operator == (const Place& p1, const Place& p2) {
  return p1.procid == p2.procid 
    && p1.device_id == p2.device_id
    && p1.device_type == p2.device_type;
}

} // end of namespace minerva
