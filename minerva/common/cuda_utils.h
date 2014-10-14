#pragma once

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <glog/logging.h>
#include <string>
#include <algorithm>

#if 0
#define CheckCudaError(msg) do { cudaDeviceSynchronize(); \
  cudaError_t e = cudaGetLastError(); \
  CHECK_EQ(e, cudaSuccess) << msg << " CUDA: " << cudaGetErrorString(e); \
} while (0)
#else
#define CheckCudaError(msg) do { \
  cudaError_t e = cudaGetLastError(); \
  CHECK_EQ(e, cudaSuccess) << msg << " CUDA: " << cudaGetErrorString(e); \
} while (0)
#endif

inline const char* CublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    default:
      break;
  }
  return "<unknown>";
}

#define CUDA_CALL(func) do { cudaError_t e = (func); \
  CHECK_EQ(e, cudaSuccess) << "CUDA: " << cudaGetErrorString(e); \
} while (0)

#define CUBLAS_CALL(func) do { cublasStatus_t e = (func); \
  CHECK_EQ(e, CUBLAS_STATUS_SUCCESS) << "CUBLAS: " << CublasGetErrorEnum(e); \
} while (0)

#define CUDNN_CALL(func) do { cudnnStatus_t e = (func); \
  CHECK_EQ(e, CUDNN_STATUS_SUCCESS); \
} while (0)

#endif

