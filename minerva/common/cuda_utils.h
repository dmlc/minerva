#pragma once

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <glog/logging.h>
#include <string>
#include <algorithm>

inline const char* CublasGetErrorString(cublasStatus_t error) {
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
  return "Unknown cuBLAS status";
}

inline const char* CudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cuDNN status";
}

inline const char* CurandGetErrorString(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown cuRAND status";
}

#define CheckCudaError(msg) { \
  cudaError_t e = cudaGetLastError(); \
  CHECK_EQ(e, cudaSuccess) << msg << " CUDA: " << cudaGetErrorString(e); \
}

#define CUDA_CALL(func) { \
  cudaError_t e = (func); \
  CHECK_EQ(e, cudaSuccess) << "CUDA: " << cudaGetErrorString(e); \
}

#define CUBLAS_CALL(func) { \
  cublasStatus_t e = (func); \
  CHECK_EQ(e, CUBLAS_STATUS_SUCCESS) << "cuBLAS: " << CublasGetErrorString(e); \
}

#define CUDNN_CALL(func) { \
  cudnnStatus_t e = (func); \
  CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << CudnnGetErrorString(e); \
}

#define CURAND_CALL(func) { \
  curandStatus_t e = (func); \
  CHECK_EQ(e, CURAND_STATUS_SUCCESS) << "cuRAND: " << CurandGetErrorString(e); \
}

#endif

