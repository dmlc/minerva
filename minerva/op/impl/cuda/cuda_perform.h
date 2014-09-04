#pragma once
#ifdef HAS_CUDA
#include <cuda_runtime.h>

namespace minerva {
namespace cuda {

void CudaPerformArithmeticAdd(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformArithmeticSub(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformArithmeticMult(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformArithmeticDiv(float*, float*, float*, size_t, cudaStream_t);

}
}
#endif
