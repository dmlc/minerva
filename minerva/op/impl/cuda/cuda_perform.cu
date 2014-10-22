#include "op/impl/cuda/cuda_kernel.h"
#include "op/impl/cuda/cuda_perform.h"
#include "common/cuda_utils.h"
#include <glog/logging.h>
#include <cublas_v2.h>
#include <limits>

namespace minerva {
namespace cuda {

static void FindConfiguration(size_t size, int& num_blocks, int& num_threads) {
  num_threads = 32;
  num_blocks = static_cast<int>((size + num_threads - 1) / num_threads);
  if (num_blocks < 0 || 128 < num_blocks) {
    num_blocks = 128;
  }
}

void CudaPerformDotMult(float* a, float* b, float* c, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(a, b, c, size, MultOp());
  CheckCudaError("CudaPerformDotMult");
}

void CudaPerformDotDiv(float* a, float* b, float* c, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(a, b, c, size, DivOp());
  CheckCudaError("CudaPerformDotDiv");
}

void CudaPerformAdd(float* a, float* b, float* c, int m, int n, cublasHandle_t handle) {
  float one = 1.0;
  CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, a, m, &one, b, m, c, m));
}

void CudaPerformSub(float* a, float* b, float* c, int m, int n, cublasHandle_t handle) {
  float minus_one = -1.0;
  float one = 1.0;
  CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, a, m, &minus_one, b, m, c, m));
}

void CudaPerformMatMult(float* a, float* b, float* c, int m, int n, int k, cublasHandle_t handle) {
  float one = 1.0;
  float zero = 0.0;
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, a, m, b, k, &zero, c, m));
}

void CudaPerformScale(float* a, float* c, int m, int n, float val, cublasHandle_t handle) {
  float zero = 0.0;
  CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &val, a, m, &zero, c, m, c, m));
}

void CudaPerformTranspose(float* a, float* c, int m, int n, cublasHandle_t handle) {
  float zero = 0.0;
  float one = 1.0;
  CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &one, a, m, &zero, c, n, c, n));
}

void CudaPerformLeftConstSub(float* in, float* out, float val, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(in, out, val, size, ReverseSubOp());
  CheckCudaError("CudaPerformLeftConstSub");
}

void CudaPerformLeftConstDiv(float* in, float* out, float val, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(in, out, val, size, ReverseDivOp());
  CheckCudaError("CudaPerformLeftConstDiv");
}

void CudaPerformNormAddOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel<<<block, thread, 0, stream>>>(matrix, row, res, m, n, SumOp());
  CheckCudaError("CudaPerformNormAddOnCol");
}

void CudaPerformNormSubOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel<<<block, thread, 0, stream>>>(matrix, row, res, m, n, SubOp());
  CheckCudaError("CudaPerformNormSubOnCol");
}

void CudaPerformNormMultOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel<<<block, thread, 0, stream>>>(matrix, row, res, m, n, MultOp());
  CheckCudaError("CudaPerformNormMultOnCol");
}

void CudaPerformNormDivOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel<<<block, thread, 0, stream>>>(matrix, row, res, m, n, DivOp());
  CheckCudaError("CudaPerformNormDivOnCol");
}

void CudaPerformNormAddOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel<<<block, thread, 0, stream>>>(matrix, row, res, m, n, SumOp());
  CheckCudaError("CudaPerformNormAddOnRow");
}

void CudaPerformNormSubOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel<<<block, thread, 0, stream>>>(matrix, row, res, m, n, SubOp());
  CheckCudaError("CudaPerformNormSubOnRow");
}

void CudaPerformNormMultOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel<<<block, thread, 0, stream>>>(matrix, row, res, m, n, MultOp());
  CheckCudaError("CudaPerformNormMultOnRow");
}

void CudaPerformNormDivOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel<<<block, thread, 0, stream>>>(matrix, row, res, m, n, DivOp());
  CheckCudaError("CudaPerformNormDivOnRow");
}

void CudaPerformReductionSumOnCol(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformReductionOnColKernel<<<block, thread, 0, stream>>>(in, out, m, n, SumOp());
  CheckCudaError("CudaPerformReductionSumOnCol");
}

void CudaPerformReductionMaxOnCol(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformReductionOnColKernel<<<block, thread, 0, stream>>>(in, out, m, n, MaxOp());
  CheckCudaError("CudaPerformReductionMaxOnCol");
}

void CudaPerformReductionSumOnRow(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformReductionOnRowKernel<<<block, thread, 0, stream>>>(in, out, m, n, SumOp());
  CheckCudaError("CudaPerformReductionSumOnRow");
}

void CudaPerformReductionMaxOnRow(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformReductionOnRowKernel<<<block, thread, 0, stream>>>(in, out, m, n, MaxOp());
  CheckCudaError("CudaPerformReductionMaxOnRow");
}

void CudaPerformMaxIndexOnCol(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformMaxIndexOnColKernel<<<block, thread, 0, stream>>>(in, out, m, n);
  CheckCudaError("CudaPerformMaxIndexOnCol");
}

void CudaPerformMaxIndexOnRow(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformMaxIndexOnRowKernel << <block, thread, 0, stream>>>(in, out, m, n);
  CheckCudaError("CudaPerformMaxIndexOnRow");
}

void CudaPerformElewiseExp(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(in, out, size, ExpOp());
  CheckCudaError("CudaPerformEleWiseExp");
}

void CudaPerformElewiseLn(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(in, out, size, LnOp());
  CheckCudaError("CudaPerformEleWiseLn");
}

void CudaPerformElewiseSigmoid(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(in, out, size, SigmoidOp());
  CheckCudaError("CudaPerformEleWiseSigmoid");
}

void CudaPerformElewiseNegative(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(in, out, size, NegativeOp());
  CheckCudaError("CudaPerformEleWiseNegative");
}

void CudaPerformConvForward(float* bottom, float* filter, float* bias, float* top, int num_images, int bottom_num_channels, int top_num_channels, int bottom_height, int bottom_width, int pad_height, int pad_width, int stride_vertical, int stride_horizontal, int filter_height, int filter_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t bottom_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnTensor4dDescriptor_t bias_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnTensor4dDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bias_desc));
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, bottom_num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetFilterDescriptor(filter_desc, CUDNN_DATA_FLOAT, top_num_channels, bottom_num_channels, filter_height, filter_width));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, top_num_channels, 1, 1));
  CUDNN_CALL(cudnnSetConvolutionDescriptor(conv_desc, bottom_desc, filter_desc, pad_height, pad_width, stride_vertical, stride_horizontal, 1, 1, CUDNN_CONVOLUTION));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, top_num_channels, (bottom_height + 2 * pad_height - filter_height) / stride_vertical + 1, (bottom_width + 2 * pad_width - filter_width) / stride_horizontal + 1));

  float one = 1;
  CUDNN_CALL(cudnnConvolutionForward(handle, bottom_desc, bottom, filter_desc, filter, conv_desc, top_desc, top, CUDNN_RESULT_NO_ACCUMULATE));
  CUDNN_CALL(cudnnAddTensor4d(handle, CUDNN_ADD_SAME_C, &one, bias_desc, bias, top_desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bias_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bottom_desc));
}

void CudaPerformConvBackwardData(float* top_diff, float* filter, float* bottom_diff, int num_images, int bottom_num_channels, int top_num_channels, int top_height, int top_width, int pad_height, int pad_width, int stride_vertical, int stride_horizontal, int filter_height, int filter_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t bottom_diff_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnTensor4dDescriptor_t top_diff_desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bottom_diff_desc));
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&top_diff_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_diff_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, bottom_num_channels, (top_height - 1) * stride_vertical + filter_height - 2 * pad_height, (top_width - 1) * stride_horizontal + filter_width - 2 * pad_width));
  CUDNN_CALL(cudnnSetFilterDescriptor(filter_desc, CUDNN_DATA_FLOAT, top_num_channels, bottom_num_channels, filter_height, filter_width));
  CUDNN_CALL(cudnnSetConvolutionDescriptor(conv_desc, bottom_diff_desc, filter_desc, pad_height, pad_width, stride_vertical, stride_horizontal, 1, 1, CUDNN_CONVOLUTION));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_diff_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, top_num_channels, top_height, top_width));

  CUDNN_CALL(cudnnConvolutionBackwardData(handle, filter_desc, filter, top_diff_desc, top_diff, conv_desc, bottom_diff_desc, bottom_diff, CUDNN_RESULT_NO_ACCUMULATE));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(top_diff_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bottom_diff_desc));
}

void CudaPerformConvBackwardFilter(float* bottom, float* top_diff, float* filter_diff, int num_images, int bottom_num_channels, int top_num_channels, int bottom_height, int bottom_width, int pad_height, int pad_width, int stride_vertical, int stride_horizontal, int filter_height, int filter_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t bottom_desc;
  cudnnFilterDescriptor_t filter_diff_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnTensor4dDescriptor_t top_diff_desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_diff_desc));
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&top_diff_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, bottom_num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetFilterDescriptor(filter_diff_desc, CUDNN_DATA_FLOAT, top_num_channels, bottom_num_channels, filter_height, filter_width));
  CUDNN_CALL(cudnnSetConvolutionDescriptor(conv_desc, bottom_desc, filter_diff_desc, pad_height, pad_width, stride_vertical, stride_horizontal, 1, 1, CUDNN_CONVOLUTION));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_diff_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, top_num_channels, (bottom_height + 2 * pad_height - filter_height) / stride_vertical + 1, (bottom_width + 2 * pad_width - filter_width) / stride_horizontal + 1));

  CUDNN_CALL(cudnnConvolutionBackwardFilter(handle, bottom_desc, bottom, top_diff_desc, top_diff, conv_desc, filter_diff_desc, filter_diff, CUDNN_RESULT_NO_ACCUMULATE));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(top_diff_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_diff_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bottom_desc));
}

void CudaPerformConvBackwardBias(float* top_diff, float* bias_diff, int num_images, int top_num_channels, int top_height, int top_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t bias_diff_desc;
  cudnnTensor4dDescriptor_t top_diff_desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bias_diff_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&top_diff_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bias_diff_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, top_num_channels, 1, 1));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_diff_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, top_num_channels, top_height, top_width));

  CUDNN_CALL(cudnnConvolutionBackwardBias(handle, top_diff_desc, top_diff, bias_diff_desc, bias_diff, CUDNN_RESULT_NO_ACCUMULATE));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(top_diff_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bias_diff_desc));
}

void CudaPerformInstanceSoftmaxForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, desc, bottom, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformChannelSoftmaxForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, desc, bottom, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformInstanceSoftmaxBackward(float* diff, float* top, float* bottom, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, desc, top, desc, diff, desc, bottom));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformChannelSoftmaxBackward(float* diff, float* top, float* bottom, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, desc, top, desc, diff, desc, bottom));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformSigmoidForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnActivationForward(handle, CUDNN_ACTIVATION_SIGMOID, desc, bottom, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformReluForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnActivationForward(handle, CUDNN_ACTIVATION_RELU, desc, bottom, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformTanhForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnActivationForward(handle, CUDNN_ACTIVATION_TANH, desc, bottom, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformSigmoidBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnActivationBackward(handle, CUDNN_ACTIVATION_SIGMOID, desc, top, desc, top_diff, desc, bottom, desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformReluBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnActivationBackward(handle, CUDNN_ACTIVATION_RELU, desc, top, desc, top_diff, desc, bottom, desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformTanhBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  CUDNN_CALL(cudnnActivationBackward(handle, CUDNN_ACTIVATION_TANH, desc, top, desc, top_diff, desc, bottom, desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(desc));
}

void CudaPerformMaxPoolingForward(float* bottom, float* top, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t bottom_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensor4dDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetPoolingDescriptor(pool_desc, CUDNN_POOLING_MAX, window_height, window_width, stride_vertical, stride_horizontal));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, (bottom_height - window_height) / stride_vertical + 1, (bottom_width - window_width) / stride_horizontal + 1));

  CUDNN_CALL(cudnnPoolingForward(handle, pool_desc, bottom_desc, bottom, top_desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bottom_desc));
}

void CudaPerformAveragePoolingForward(float* bottom, float* top, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t bottom_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensor4dDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetPoolingDescriptor(pool_desc, CUDNN_POOLING_AVERAGE, window_height, window_width, stride_vertical, stride_horizontal));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, (bottom_height - window_height) / stride_vertical + 1, (bottom_width - window_width) / stride_horizontal + 1));

  CUDNN_CALL(cudnnPoolingForward(handle, pool_desc, bottom_desc, bottom, top_desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bottom_desc));
}

void CudaPerformMaxPoolingBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t bottom_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensor4dDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetPoolingDescriptor(pool_desc, CUDNN_POOLING_MAX, window_height, window_width, stride_vertical, stride_horizontal));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, (bottom_height - window_height) / stride_vertical + 1, (bottom_width - window_width) / stride_horizontal + 1));

  CUDNN_CALL(cudnnPoolingBackward(handle, pool_desc, top_desc, top, top_desc, top_diff, bottom_desc, bottom, bottom_desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bottom_desc));
}

void CudaPerformAveragePoolingBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensor4dDescriptor_t bottom_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensor4dDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnCreateTensor4dDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetPoolingDescriptor(pool_desc, CUDNN_POOLING_AVERAGE, window_height, window_width, stride_vertical, stride_horizontal));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, (bottom_height - window_height) / stride_vertical + 1, (bottom_width - window_width) / stride_horizontal + 1));

  CUDNN_CALL(cudnnPoolingBackward(handle, pool_desc, top_desc, top, top_desc, top_diff, bottom_desc, bottom, bottom_desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensor4dDescriptor(bottom_desc));
}

}  // namespace cuda
}  // namespace minerva

