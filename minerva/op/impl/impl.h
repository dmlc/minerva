#pragma once

namespace minerva {

enum IMPL_TYPE {
  BASIC = 0,
  MKL,
  CUDA,
};

template<class C>
class FnBundle {
};

#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(DataList& i, DataList& o, closure_name& c, IMPL_TYPE it) {\
      switch(it) {\
        case BASIC: basic_fn(i, o, c); break;\
        case MKL: mkl_fn(i, o, c); break;\
        case CUDA: cuda_fn(i, o, c); break;\
        default: NO_IMPL(i, o, c); break;\
      }\
    }\
  };

#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(DataShard& d, closure_name& c, IMPL_TYPE it) {\
      switch(it) {\
        case BASIC: basic_fn(d, c); break;\
        case MKL: mkl_fn(d, c); break;\
        case CUDA: cuda_fn(d, c); break;\
        default: NO_IMPL(d, c); break;\
      }\
    }\
  };

}
