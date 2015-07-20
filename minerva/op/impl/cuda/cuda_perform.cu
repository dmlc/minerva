#include "op/impl/cuda/cuda_kernel.h"
#include "op/impl/cuda/cuda_perform.h"
#include "./cuda_perform_system_include.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <limits>

namespace minerva {
namespace cuda {

static void FindConfiguration(size_t size, int& num_blocks, int& num_threads) {
  num_threads = 0;
  if(size <= 32)
    num_threads = 32;
  else if(size <= 64)
    num_threads = 64;
  else if(size <= 128)
    num_threads = 128;
  else if(size <= 256)
    num_threads = 256;
  else if(size <= 512)
    num_threads = 512;
  else
    num_threads = 1024;
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

void CudaPerformAdd(float* a, float* b, float* c, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(a, b, c, size, SumOp());
  CheckCudaError("CudaPerformAdd");
  //float one = 1.0;
  //CUBLAS_CALL(cublasScopy(handle, size, a, 1, c, 1));
  //CUBLAS_CALL(cublasSaxpy(handle, size, &one, b, 1, c, 1));
}

void CudaPerformCopy(float* a, float* b, size_t size, cublasHandle_t handle) {
  CUBLAS_CALL(cublasScopy(handle, size, a, 1, b, 1));
}

void CudaPerformSub(float* a, float* b, float* c, size_t size, cublasHandle_t handle) {
  float minus_one = -1.0;
  CUBLAS_CALL(cublasScopy(handle, size, a, 1, c, 1));
  CUBLAS_CALL(cublasSaxpy(handle, size, &minus_one, b, 1, c, 1));
}

void CudaPerformMatMult(float* a, float* b, float* c, int m, int n, int k, cublasHandle_t handle) {
  float one = 1.0;
  float zero = 0.0;
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, a, m, b, k, &zero, c, m));
}

void CudaPerformScale(float* in_data, float* res_data, size_t size, float val, cublasHandle_t handle) {
  CUBLAS_CALL(cublasScopy(handle, size, in_data, 1, res_data, 1));
  CUBLAS_CALL(cublasSscal(handle, size, &val, res_data, 1));
}

void CudaPerformTranspose(float* a, float* c, int m, int n, cublasHandle_t handle) {
  float zero = 0.0;
  float one = 1.0;
  CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &one, a, m, &zero, c, n, c, n));
}

void CudaPerformConstAdd(float* in, float* out, float val, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(in, out, val, size, SumOp());
  CheckCudaError("CudaPerformConstAdd");
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

void CudaPerformReshape(float* in, float* out, size_t size, cudaStream_t stream) {
  CUDA_CALL(cudaMemcpyAsync(out, in, size, cudaMemcpyDefault, stream));
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

void CudaPerformElewiseNegative(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel<<<block, thread, 0, stream>>>(in, out, size, NegativeOp());
  CheckCudaError("CudaPerformEleWiseNegative");
}

void CudaPerformConvBackwardBias(float* top_diff, float* bias_diff, int num_images, int top_num_channels, int top_height, int top_width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t bias_diff_desc;
  cudnnTensorDescriptor_t top_diff_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_diff_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&top_diff_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bias_diff_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, top_num_channels, 1, 1));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_diff_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, top_num_channels, top_height, top_width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnConvolutionBackwardBias(handle, &one, top_diff_desc, top_diff, &zero, bias_diff_desc, bias_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(top_diff_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_diff_desc));
}

void CudaPerformInstanceSoftmaxForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &one, desc, bottom, &zero, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformChannelSoftmaxForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one, desc, bottom, &zero, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformInstanceSoftmaxBackward(float* diff, float* top, float* bottom, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &one, desc, top, desc, diff, &zero, desc, bottom));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformChannelSoftmaxBackward(float* diff, float* top, float* bottom, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one, desc, top, desc, diff, &zero, desc, bottom));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformSigmoidForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnActivationForward(handle, CUDNN_ACTIVATION_SIGMOID, &one, desc, bottom, &zero, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformReluForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnActivationForward(handle, CUDNN_ACTIVATION_RELU, &one, desc, bottom, &zero, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformTanhForward(float* bottom, float* top, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnActivationForward(handle, CUDNN_ACTIVATION_TANH, &one, desc, bottom, &zero, desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformSigmoidBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnActivationBackward(handle, CUDNN_ACTIVATION_SIGMOID, &one, desc, top, desc, top_diff, desc, bottom, &zero, desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformReluBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnActivationBackward(handle, CUDNN_ACTIVATION_RELU, &one, desc, top, desc, top_diff, desc, bottom, &zero, desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformTanhBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int height, int width, cudaStream_t stream, cudnnHandle_t handle) {
  cudnnTensorDescriptor_t desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, height, width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnActivationBackward(handle, CUDNN_ACTIVATION_TANH, &one, desc, top, desc, top_diff, desc, bottom, &zero, desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
}

void CudaPerformMaxPoolingForward(float* bottom, float* top, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, int pad_height, int pad_width, cudaStream_t stream, cudnnHandle_t handle) {
  // Calculate the dimension after pooling
  int pooled_height = (bottom_height + 2 * pad_height - window_height + stride_vertical - 1) / stride_vertical + 1;
  if (0 <= (pooled_height - 1) * stride_vertical - bottom_height - pad_height) {
    --pooled_height;
  }
  int pooled_width = (bottom_width + 2 * pad_width - window_width + stride_horizontal - 1) / stride_horizontal + 1;
  if (0 <= (pooled_width - 1) * stride_horizontal - bottom_width - pad_width) {
    --pooled_width;
  }
  cudnnTensorDescriptor_t bottom_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensorDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, window_height, window_width, pad_height, pad_width, stride_vertical, stride_horizontal));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, pooled_height, pooled_width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnPoolingForward(handle, pool_desc, &one, bottom_desc, bottom, &zero, top_desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bottom_desc));
}

void CudaPerformAveragePoolingForward(float* bottom, float* top, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, int pad_height, int pad_width, cudaStream_t stream, cudnnHandle_t handle) {
  // Calculate the dimension after pooling
  int pooled_height = (bottom_height + 2 * pad_height - window_height + stride_vertical - 1) / stride_vertical + 1;
  if (0 <= (pooled_height - 1) * stride_vertical - bottom_height - pad_height) {
    --pooled_height;
  }
  int pooled_width = (bottom_width + 2 * pad_width - window_width + stride_horizontal - 1) / stride_horizontal + 1;
  if (0 <= (pooled_width - 1) * stride_horizontal - bottom_width - pad_width) {
    --pooled_width;
  }
  cudnnTensorDescriptor_t bottom_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensorDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, window_height, window_width, pad_height, pad_width, stride_vertical, stride_horizontal));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, pooled_height, pooled_width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnPoolingForward(handle, pool_desc, &one, bottom_desc, bottom, &zero, top_desc, top));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bottom_desc));
}

void CudaPerformMaxPoolingBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, int pad_height, int pad_width, cudaStream_t stream, cudnnHandle_t handle) {
  // Calculate the dimension after pooling
  int pooled_height = (bottom_height + 2 * pad_height - window_height + stride_vertical - 1) / stride_vertical + 1;
  if (0 <= (pooled_height - 1) * stride_vertical - bottom_height - pad_height) {
    --pooled_height;
  }
  int pooled_width = (bottom_width + 2 * pad_width - window_width + stride_horizontal - 1) / stride_horizontal + 1;
  if (0 <= (pooled_width - 1) * stride_horizontal - bottom_width - pad_width) {
    --pooled_width;
  }
  cudnnTensorDescriptor_t bottom_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensorDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, window_height, window_width, pad_height, pad_width, stride_vertical, stride_horizontal));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, pooled_height, pooled_width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnPoolingBackward(handle, pool_desc, &one, top_desc, top, top_desc, top_diff, bottom_desc, bottom, &zero, bottom_desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bottom_desc));
}

void CudaPerformAveragePoolingBackward(float* bottom, float* top, float* top_diff, float* bottom_diff, int num_images, int num_channels, int bottom_height, int bottom_width, int stride_vertical, int stride_horizontal, int window_height, int window_width, int pad_height, int pad_width, cudaStream_t stream, cudnnHandle_t handle) {
  // Calculate the dimension after pooling
  int pooled_height = (bottom_height + 2 * pad_height - window_height + stride_vertical - 1) / stride_vertical + 1;
  if (0 <= (pooled_height - 1) * stride_vertical - bottom_height - pad_height) {
    --pooled_height;
  }
  int pooled_width = (bottom_width + 2 * pad_width - window_width + stride_horizontal - 1) / stride_horizontal + 1;
  if (0 <= (pooled_width - 1) * stride_horizontal - bottom_width - pad_width) {
    --pooled_width;
  }
  cudnnTensorDescriptor_t bottom_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnTensorDescriptor_t top_desc;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&bottom_desc));
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&top_desc));

  CUDNN_CALL(cudnnSetTensor4dDescriptor(bottom_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, bottom_height, bottom_width));
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, window_height, window_width, pad_height, pad_width, stride_vertical, stride_horizontal));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(top_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_images, num_channels, pooled_height, pooled_width));

  float one = 1;
  float zero = 0;
  CUDNN_CALL(cudnnPoolingBackward(handle, pool_desc, &one, top_desc, top, top_desc, top_diff, bottom_desc, bottom, &zero, bottom_desc, bottom_diff));
  CUDA_CALL(cudaStreamSynchronize(stream));  // Synchronize before destruction

  CUDNN_CALL(cudnnDestroyTensorDescriptor(top_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bottom_desc));
}

void CudaPerformRandn(float* dst, size_t size, unsigned int seed, float mean, float var) {
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
  CURAND_CALL(curandGenerateNormal(gen, dst, size, mean, var));
  CURAND_CALL(curandDestroyGenerator(gen));
}

void CudaPerformRandBernoulli(float* dst, size_t size, unsigned int seed, float p, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformRandBernoulliKernel<<<block, thread, 0, stream>>>(dst, size, seed, p);
  CheckCudaError(__func__);
}

void CudaPerformFill(float* dst, size_t size, float val, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformFillKernel<<<block, thread, 0, stream>>>(dst, size, val);
  CheckCudaError("CudaPerformFill");
}

void CudaPerformSelect(float* dst, float* src, std::vector<int> indices, size_t cols, size_t rows, cudaStream_t stream) {
  int block, thread;
  int size = cols * rows;
  int* indices_ptr;
  FindConfiguration(size, block, thread);
  CUDA_CALL(cudaMalloc(&indices_ptr, indices.size() * sizeof(int)));
  CUDA_CALL(cudaMemcpyAsync(indices_ptr, &indices[0], indices.size() * sizeof(int), cudaMemcpyDefault, stream));
  SelectKernel<<<block, thread, 0, stream>>>(dst, src, &indices[0], cols, rows, indices.size());
  CheckCudaError("Select");
}

}  // namespace cuda
}  // namespace minerva

