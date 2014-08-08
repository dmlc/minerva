#pragma once
#include <iostream>

namespace minerva {

enum class IMPL_TYPE {
  NA = 0,
  BASIC,
  MKL,
  CUDA,
};

inline std::ostream& operator << (std::ostream& os, IMPL_TYPE t) {
  switch(t) {
    case IMPL_TYPE::NA: return os << "N/A";
    case IMPL_TYPE::BASIC: return os << "Basic";
    case IMPL_TYPE::MKL: return os << "Mkl";
    case IMPL_TYPE::CUDA: return os << "Cuda";
    default: return os << "Unknown impl type";
  }
}

template<class C>
class FnBundle {
};

#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(DataList& i, DataList& o, closure_name& c, IMPL_TYPE it) {\
      switch(it) {\
        case IMPL_TYPE::BASIC: basic_fn(i, o, c); break;\
        case IMPL_TYPE::MKL: mkl_fn(i, o, c); break;\
        case IMPL_TYPE::CUDA: cuda_fn(i, o, c); break;\
        default: NO_IMPL(i, o, c); break;\
      }\
    }\
  };

#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(DataList& d, closure_name& c, IMPL_TYPE it) {\
      switch(it) {\
        case IMPL_TYPE::BASIC: basic_fn(d, c); break;\
        case IMPL_TYPE::MKL: mkl_fn(d, c); break;\
        case IMPL_TYPE::CUDA: cuda_fn(d, c); break;\
        default: NO_IMPL(d, c); break;\
      }\
    }\
  };

}
