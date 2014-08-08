#pragma once
#include <iostream>

namespace minerva {

enum class ImplType {
  NA = 0,
  BASIC,
  MKL,
  CUDA,
};

inline std::ostream& operator << (std::ostream& os, ImplType t) {
  switch(t) {
    case ImplType::NA: return os << "N/A";
    case ImplType::BASIC: return os << "Basic";
    case ImplType::MKL: return os << "Mkl";
    case ImplType::CUDA: return os << "Cuda";
    default: return os << "Unknown impl type";
  }
}

template<class C>
class FnBundle {
};

#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(DataList& i, DataList& o, closure_name& c, ImplType it) {\
      switch(it) {\
        case ImplType::BASIC: basic_fn(i, o, c); break;\
        case ImplType::MKL: mkl_fn(i, o, c); break;\
        case ImplType::CUDA: cuda_fn(i, o, c); break;\
        default: NO_IMPL(i, o, c); break;\
      }\
    }\
  };

#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(DataList& d, closure_name& c, ImplType it) {\
      switch(it) {\
        case ImplType::BASIC: basic_fn(d, c); break;\
        case ImplType::MKL: mkl_fn(d, c); break;\
        case ImplType::CUDA: cuda_fn(d, c); break;\
        default: NO_IMPL(d, c); break;\
      }\
    }\
  };

}
