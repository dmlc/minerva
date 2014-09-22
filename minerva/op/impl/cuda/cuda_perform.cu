#include "op/impl/cuda/cuda_perform.h"
#include "op/op_types.h"
#include "common/cuda_utils.h"
#include <glog/logging.h>
#include <cublas_v2.h>
#include <limits>

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

template<class function>
__global__ static void CudaPerformPerElementKernel(float* in, float* out, float v, size_t size, function func)
{
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    func(in + cur, out + cur, v);
    cur += gridDim.x * blockDim.x;
  }
}

__global__ static void CudaPerformNormOnColKernel(float* matrix, float* row, float * res,
  int m, int n, minerva::ArithmeticType type)
{
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  switch (type)
  {
  case minerva::ADD:
    while (row_id < m)
    {
      float* mat = matrix + row_id;
      for (int i = 0; i < n; i++)
      {
        res[i*n + row_id] = mat[i*n + row_id] + row[i];
      }
      row_id += step;
    }
    break;
  case minerva::SUB:
    while (row_id < m)
    {
      float* mat = matrix + row_id;
      for (int i = 0; i < n; i++)
      {
        res[i*n + row_id] = mat[i*n + row_id] - row[i];
      }
      row_id += step;
    }
    break;
  case minerva::MULT:
    while (row_id < m)
    {
      float* mat = matrix + row_id;
      for (int i = 0; i < n; i++)
      {
        res[i*n + row_id] = mat[i*n + row_id] * row[i];
      }
      row_id += step;
    }
    break;
  case minerva::DIV:
    while (row_id < m)
    {
      float* mat = matrix + row_id;
      for (int i = 0; i < n; i++)
      {
        res[i*n + row_id] = mat[i*n + row_id] / row[i];
      }
      row_id += step;
    }
    break;
  }  
}

__global__ static void CudaPerformNormOnRowKernel(float* matrix, float* col, float * res,
  int m, int n, minerva::ArithmeticType type)
{
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  switch (type)
  {
  case minerva::ADD:
    while (row_id < m)
    {
      float* mat = matrix + row_id;
      for (int i = 0; i < n; i++)
      {
        res[i*n + row_id] = mat[i*n + row_id] + col[row_id];
      }
      row_id += step;
    }
    break;
  case minerva::SUB:
    while (row_id < m)
    {
      float* mat = matrix + row_id;
      for (int i = 0; i < n; i++)
      {
        res[i*n + row_id] = mat[i*n + row_id] - col[row_id];
      }
      row_id += step;
    }
    break;
  case minerva::MULT:
    while (row_id < m)
    {
      float* mat = matrix + row_id;
      for (int i = 0; i < n; i++)
      {
        res[i*n + row_id] = mat[i*n + row_id] * col[row_id];
      }
      row_id += step;
    }
    break;
  case minerva::DIV:
    while (row_id < m)
    {
      float* mat = matrix + row_id;
      for (int i = 0; i < n; i++)
      {
        res[i*n + row_id] = mat[i*n + row_id] / col[row_id];
      }
      row_id += step;
    }
    break;
  }
}

namespace minerva {
namespace cuda {

  static void FindConfiguration(size_t size, int & num_blocks, int & num_threads)
  {
    num_threads = 32;
    num_blocks = int((size + num_threads - 1)/ num_threads);
    if (num_blocks < 0 || num_blocks > 128)
    {
      num_blocks = 128;
    }
  }

void CudaPerformDotMult(float* a, float* b, float* c, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformArithmeticMultKernel <<<block, thread, 0, stream >> >(c, a, b, size);
  CheckCudaError("CudaPerformDotMult");
}

void CudaPerformDotDiv(float* a, float* b, float* c, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformArithmeticDivKernel <<<block, thread, 0, stream >> >(c, a, b, size);
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

void CudaPerformTranspose(float* a, float* c, int m, int n, cublasHandle_t handle)
{
  float zero = 0.0;
  float one = 1.0;
  CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &one, a, m, &zero, c, n, c, n));
}

class LeftConstSubOp
{
public:
  __device__ inline void operator()(float* in, float* out, float v) const
  {
    *out = v - *in;
  }
};

void CudaPerformLeftConstSub(float* in, float* out, float val, size_t size, cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(size, block, thread);
  LeftConstSubOp op;
  CudaPerformPerElementKernel <<<block, thread, 0, stream >> >(in, out, val, size, op);
  CheckCudaError("CudaPerformLeftConstSub");
}

class LeftConstDivOp
{
public:
  __device__ inline void operator()(float* in, float* out, float v) const
  {
    *out = v/(*in);
  }
};

void CudaPerformLeftConstDiv(float* in, float* out, float val, size_t size, cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(size, block, thread);
  LeftConstDivOp op;
  CudaPerformPerElementKernel <<<block, thread, 0, stream >> >(in, out, val, size, op);
  CheckCudaError("CudaPerformLeftConstDiv");
}

void CudaPerformNormOnCol(float* matrix, float* row, float* res, int m, int n,
  const ArithmeticType & type, cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel <<<block, thread, 0, stream >> >(matrix, row, res, m, n, type);
  CheckCudaError("CudaPerformNormOnCol");
}

void CudaPerformNormOnRow(float* matrix, float* col, float* res, int m, int n,
  const ArithmeticType & type, cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel <<<block, thread, 0, stream >> >(matrix, col, res, m, n, type);
  CheckCudaError("CudaPerformNormOnRow");
}

class EXPOp
{
public:
  __device__ inline void operator()(float* in, float* out, float v) const
  {
    *out = expf(*in);
  }
};

class LNOp
{
public:
  __device__ inline void operator()(float* in, float* out, float v) const
  {
    *out = log(*in);
  }
};

class NegativeOp
{
public:
  __device__ inline void operator()(float* in, float* out, float v) const
  {
    *out = -(*in);
  }
};

class SigmoidOp
{
public:
  __device__ inline void operator()(float* in, float* out, float v) const
  {
    float x = *in;
#if 0
    *out = 1 / (1 + exp(-x));
#else
    *out = x / (1 + abs(x));
#endif
  }
};

void CudaPerformEleWise(float* in, float* out, size_t size, 
  const ElewiseType & type, cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(size, block, thread);
  switch (type)
  {
  case EXP:
    EXPOp expop;
    CudaPerformPerElementKernel << <block, thread, 0, stream >> >(in, out, 0, size, expop);
    break;
  case LN:
    LNOp lnop;
    CudaPerformPerElementKernel << <block, thread, 0, stream >> >(in, out, 0, size, lnop);
    break;
  case SIGMOID:
    SigmoidOp sigmoidop;
    CudaPerformPerElementKernel << <block, thread, 0, stream >> >(in, out, 0, size, sigmoidop);
    break;
  case NEGATIVE:
    NegativeOp negop;
    CudaPerformPerElementKernel << <block, thread, 0, stream >> >(in, out, 0, size, negop);
    break;
  }
  
  CheckCudaError("CudaPerformLeftConstSub");
}





}
}
