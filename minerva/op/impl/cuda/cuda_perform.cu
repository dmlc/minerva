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

void CudaPerformMatMult(float* a, float* b, float* c, size_t m, size_t k, size_t n, cublasHandle_t handle, float* zero, float* one) {
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, one, a, m, b, k, zero, c, m);
}

}
}
