#pragma once
#include <iostream>

namespace minerva {

enum class ImplType {
  kNA = 0,
  kBasic,
  kMkl,
  kCuda,
};

inline std::ostream& operator << (std::ostream& os, ImplType t) {
  switch(t) {
    case ImplType::kNA: return os << "N/A";
    case ImplType::kBasic: return os << "Basic";
    case ImplType::kMkl: return os << "Mkl";
    case ImplType::kCuda: return os << "Cuda";
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
        case ImplType::kBasic: basic_fn(i, o, c); break;\
        case ImplType::kMkl: mkl_fn(i, o, c); break;\
        case ImplType::kCuda: cuda_fn(i, o, c); break;\
        default: NO_IMPL(i, o, c); break;\
      }\
    }\
  };

#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(DataList& d, closure_name& c, ImplType it) {\
      switch(it) {\
        case ImplType::kBasic: basic_fn(d, c); break;\
        case ImplType::kMkl: mkl_fn(d, c); break;\
        case ImplType::kCuda: cuda_fn(d, c); break;\
        default: NO_IMPL(d, c); break;\
      }\
    }\
  };

}
