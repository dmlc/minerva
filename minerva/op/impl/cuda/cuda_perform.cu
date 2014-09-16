#include "op/impl/cuda/cuda_perform.h"
#include <cublas_v2.h>

__global__ static void CudaPerformArithmeticAddKernel(float* res, float* left, float* right, size_t size) {
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    *(res + cur) = *(left + cur) + *(right + cur);
    cur += gridDim.x * blockDim.x;
  }
}

__global__ static void CudaPerformArithmeticSubKernel(float* res, float* left, float* right, size_t size) {
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    *(res + cur) = *(left + cur) - *(right + cur);
    cur += gridDim.x * blockDim.x;
  }
}

__global__ static void CudaPerformArithmeticMultKernel(float* res, float* left, float* right, size_t size) {
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    *(res + cur) = *(left + cur) * *(right + cur);
    cur += gridDim.x * blockDim.x;
  }
}

__global__ static void CudaPerformArithmeticDivKernel(float* res, float* left, float* right, size_t size) {
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    *(res + cur) = *(left + cur) / *(right + cur);
    cur += gridDim.x * blockDim.x;
  }
}

namespace minerva {
namespace cuda {

void CudaPerformArithmeticAdd(float* res, float* left, float* right, size_t size, cudaStream_t stream) {
  CudaPerformArithmeticAddKernel<<<16, 16, 0, stream>>>(res, left, right, size);
}

void CudaPerformArithmeticSub(float* res, float* left, float* right, size_t size, cudaStream_t stream) {
  CudaPerformArithmeticSubKernel<<<16, 16, 0, stream>>>(res, left, right, size);
}

void CudaPerformArithmeticMult(float* res, float* left, float* right, size_t size, cudaStream_t stream) {
  CudaPerformArithmeticMultKernel<<<16, 16, 0, stream>>>(res, left, right, size);
}

void CudaPerformArithmeticDiv(float* res, float* left, float* right, size_t size, cudaStream_t stream) {
  CudaPerformArithmeticDivKernel<<<16, 16, 0, stream>>>(res, left, right, size);
}

void CudaPerformSub(float* a, float* b, float* c, int m, int n, cublasHandle_t handle) {
  float minus_one = -1.0;
  float one = 1.0;
  cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, a, m, &minus_one, b, m, c, m);
}

void CudaPerformMatMult(float* a, float* b, float* c, int m, int n, int k, cublasHandle_t handle) {
  float one = 1.0;
  float zero = 0.0;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, a, m, b, k, &zero, c, m);
}

void CudaPerformScale(float* a, float* c, int m, int n, float val, cublasHandle_t handle) {
  float zero = 0.0;
  cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &val, a, m, &zero, c, m, c, m);
}

void CudaPerformTranspose(float* a, float* c, int m, int n, cublasHandle_t handle)
{
  float zero = 0.0;
  float one = 1.0;
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &one, a, m, &zero, c, n, c, n);
}


}
}
