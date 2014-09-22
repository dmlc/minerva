#pragma once
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "op/op_types.h"

namespace minerva {
namespace cuda {

void CudaPerformDotMult(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformDotDiv(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformAdd(float* a, float* b, float* c, int m, int n, cublasHandle_t);
void CudaPerformSub(float* a, float* b, float* c, int m, int n, cublasHandle_t);
void CudaPerformMatMult(float*, float*, float*, int, int, int, cublasHandle_t);
void CudaPerformScale(float* in_data, float* res_data, int m, int n, float val, cublasHandle_t);
void CudaPerformTranspose(float* a, float* c, int m, int n, cublasHandle_t);

void CudaPerformLeftConstSub(float* in, float* out, float val, size_t, cudaStream_t);
void CudaPerformLeftConstDiv(float* in, float* out, float val, size_t, cudaStream_t);

void CudaPerformNormOnCol(float* matrix, float* row, float* res, int m, int n,
  const ArithmeticType & type, cudaStream_t);
void CudaPerformNormOnRow(float* matrix, float* col, float* res, int m, int n,
  const ArithmeticType & type, cudaStream_t);

void CudaPerformEleWise(float* in, float* out, size_t size,
  const ElewiseType & type, cudaStream_t);

}
}
#endif
