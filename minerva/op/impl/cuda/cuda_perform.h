#pragma once
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace minerva {
#ifdef HAS_CUDA
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

void CudaPerformNormAddOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t);
void CudaPerformNormSubOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t);
void CudaPerformNormMultOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t);
void CudaPerformNormDivOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t);

void CudaPerformNormAddOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t);
void CudaPerformNormSubOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t);
void CudaPerformNormMultOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t);
void CudaPerformNormDivOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t);

void CudaPerformReductionSumOnCol(float* in, float* out, int m, int n, cudaStream_t);
void CudaPerformReductionMaxOnCol(float* in, float* out, int m, int n, cudaStream_t);
void CudaPerformReductionSumOnRow(float* in, float* out, int m, int n, cudaStream_t);
void CudaPerformReductionMaxOnRow(float* in, float* out, int m, int n, cudaStream_t);

void CudaPerformMaxIndexOnCol(float* in, float* out, int m, int n, cudaStream_t);
void CudaPerformMaxIndexOnRow(float* in, float* out, int m, int n, cudaStream_t);

void CudaPerformElewiseExp(float* in, float* out, size_t size, cudaStream_t);
void CudaPerformElewiseLn(float* in, float* out, size_t size, cudaStream_t);
void CudaPerformElewiseSigmoid(float* in, float* out, size_t size, cudaStream_t);
void CudaPerformElewiseNegative(float* in, float* out, size_t size, cudaStream_t);

void CudaPerformConvForward(float* img, float* filter, float* bias, float* out, int pad_height, int pad_width, int stride_vertical, int stride_horizontal, int num_images, int num_inputs, int num_outputs, int filter_height, int filter_width);

}
#endif
}
