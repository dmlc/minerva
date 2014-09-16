#pragma once
#include <iostream>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace minerva {

enum class ImplType {
  kNA = 0,
  kBasic,
  kMkl,
  kCuda
};

inline std::ostream& operator<<(std::ostream& os, ImplType t) {
  switch (t) {
    case ImplType::kNA: return os << "N/A";
    case ImplType::kBasic: return os << "Basic";
    case ImplType::kMkl: return os << "Mkl";
    case ImplType::kCuda: return os << "Cuda";
    default: return os << "Unknown impl type";
  }
}

struct Context {
  ImplType impl_type;
  virtual ~Context() {
  };
};

struct CudaRuntimeContext : public Context {
#ifdef HAS_CUDA
  cudaStream_t stream;
#endif
};

}

