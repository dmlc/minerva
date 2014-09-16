#pragma once
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace minerva {
namespace cuda {

void CudaPerformArithmeticAdd(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformArithmeticSub(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformArithmeticMult(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformArithmeticDiv(float*, float*, float*, size_t, cudaStream_t);


void CudaPerformSub(float* a, float* b, float* c, int m, int n, cublasHandle_t);
void CudaPerformMatMult(float*, float*, float*, int, int, int, cublasHandle_t);
void CudaPerformScale(float* in_data, float* res_data, int m, int n, float val, cublasHandle_t);
void CudaPerformTranspose(float* a, float* c, int m, int n, cublasHandle_t);

}
}
#endif
