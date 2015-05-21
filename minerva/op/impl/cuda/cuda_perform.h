#pragma once

#ifdef HAS_CUDA
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

namespace minerva {
namespace cuda {

void CudaPerformDotMult(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformDotDiv(float*, float*, float*, size_t, cudaStream_t);
void CudaPerformAdd(float* a, float* b, float* c, size_t, cudaStream_t);
void CudaPerformCopy(float* a, float* b, size_t, cublasHandle_t);
void CudaPerformSub(float* a, float* b, float* c, size_t, cublasHandle_t);
void CudaPerformMatMult(float*, float*, float*, int, int, int, cublasHandle_t);
void CudaPerformScale(float* in_data, float* res_data, size_t, float val, cublasHandle_t);
void CudaPerformTranspose(float* a, float* c, int m, int n, cublasHandle_t);

void CudaPerformConstAdd(float* in, float* out, float val, size_t, cudaStream_t);
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

void CudaPerformReshape(float* in, float* out, size_t size, cudaStream_t);

void CudaPerformElewiseExp(float* in, float* out, size_t size, cudaStream_t);
void CudaPerformElewiseLn(float* in, float* out, size_t size, cudaStream_t);
void CudaPerformElewiseNegative(float* in, float* out, size_t size, cudaStream_t);

void CudaPerformConvForward(float* bottom, float* filter, float* bias, float* top, int num_images, int bottom_num_channels, int top_num_channels, int bottom_height, int bottom_width, int pad_height, int pad_width, int stride_vertical, int stride_horizontal, int filter_height, int filter_width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformConvBackwardData(float* top_diff, float* filter, float* bottom_diff, int num_images, int bottom_num_channels, int top_num_channels, int top_height, int top_width, int pad_height, int pad_width, int stride_vertical, int stride_horizontal, int filter_height, int filter_width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformConvBackwardFilter(float* bottom, float* top_diff, float* filter_diff, int num_images, int bottom_num_channels, int top_num_channels, int bottom_height, int bottom_width, int pad_height, int pad_width, int stride_vertical, int stride_horizontal, int filter_height, int filter_width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformConvBackwardBias(float* top_diff, float* bias_diff, int num_images, int top_num_channels, int top_height, int top_width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformInstanceSoftmaxForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformChannelSoftmaxForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformInstanceSoftmaxBackward(float* top_diff, float* top, float* bottom, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformChannelSoftmaxBackward(float* top_diff, float* top, float* bottom, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformSigmoidForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformReluForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformTanhForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformSigmoidBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformReluBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformTanhBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformMaxPoolingForward(float* bottom, float* top, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, int pad_height, int pad_width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformAveragePoolingForward(float* bottom, float* top, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, int pad_height, int pad_width,cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformMaxPoolingBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width,int pad_height, int pad_width, cudaStream_t stream, cudnnHandle_t handle);
void CudaPerformAveragePoolingBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, int pad_height, int pad_width, cudaStream_t stream, cudnnHandle_t handle);

void CudaPerformRandn(float* dst, size_t size, unsigned int seed, float mean, float var);
void CudaPerformRandBernoulli(float* dst, size_t size, unsigned int seed, float p, cudaStream_t stream);
void CudaPerformFill(float* dst, size_t size, float val, cudaStream_t stream);

void CudaPerformLRNForward(float* bottom, float* scale, float* res, int local_size, float alpha, float beta, int num_img, int channel, int width, int height, cudaStream_t);

void CudaPerformLRNBackward(float* bottom_data, float* top_data, float* scale, float* top_diff, float* bottom_diff, int local_size, float alpha, float beta, int num_img, int channel, int width, int height, cudaStream_t stream);

void CudaPerformSelect(float* dst, float* src, std::vector<int> indices, size_t cols, size_t rows, cudaStream_t );

} // end of namespace cuda
} // end of namespace minerva

#endif
