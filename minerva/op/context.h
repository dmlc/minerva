#pragma once
#include <iostream>
#include <functional>
#include "device/data_store.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cublas.h>
#include <cudnn.h>
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
  using TemporarySpaceAllocator =
    std::function<std::unique_ptr<TemporarySpaceHolder>(size_t)>;
  ImplType impl_type;
#ifdef HAS_CUDA
  cudaStream_t stream;
  cublasHandle_t cublas_handle;
  cudnnHandle_t cudnn_handle;
#endif
  TemporarySpaceAllocator temporary_space_allocator;
  virtual ~Context() = default;
};

}

