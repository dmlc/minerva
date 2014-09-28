#include "op/impl/cuda/cuda_perform.h"
#include "common/cuda_utils.h"
#include <glog/logging.h>
#include <cublas_v2.h>
#include <limits>

// c = a DotOp b
template<class binary_function>
__global__ static void CudaPerformDotKernel(float* a, float* b, float* c,
  size_t size, binary_function func) {
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    *(c + cur) = func(*(a + cur), *(b + cur));
    cur += gridDim.x * blockDim.x;
  }
}

// y = x Op v
template<class binary_function>
__global__ static void CudaPerformDotKernel(float* x, float* y, float v,
  size_t size, binary_function func)
{
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    y[cur] = func(x[cur], v);
    cur += gridDim.x * blockDim.x;
  }
}

// y = Op (x)
template<class binary_function>
__global__ static void CudaPerformDotKernel(float* x, float* y,
  size_t size, binary_function func)
{
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    y[cur] = func(x[cur]);
    cur += gridDim.x * blockDim.x;
  }
}

// res = matrix Norm row
template<class binary_function>
__global__ static void CudaPerformNormOnColKernel(float* matrix, float* row, float * res,
  int m, int n, binary_function func)
{
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (row_id < m)
  {
    for (int i = 0; i < n; i++)
    {
      res[i*m + row_id] = func(matrix[i*m + row_id], row[i]);
    }
    row_id += step;
  }
}

// res = matrix Norm col
template<class binary_function>
__global__ static void CudaPerformNormOnRowKernel(float* matrix, float* col, float * res,
  int m, int n, binary_function func)
{
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (row_id < m)
  {
    for (int i = 0; i < n; i++)
    {
      res[i*m + row_id] = func(matrix[i*m + row_id], col[row_id]);
    }
    row_id += step;
  }
}

// row = ReductionOp(matrix)
template<class binary_function>
__global__ static void CudaPerformReductionOnColKernel(float* matrix, float* row,
  int m, int n, binary_function func)
{
  // TODO: this is inefficient
  int col_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (col_id < n)
  {
    float r = matrix[col_id * m];
    for (int i = 0; i < m; i++)
    {
      r = func(r, matrix[col_id * m + i]);
    }
    row[col_id] = r;
    col_id += step;
  }
}

// col = ReductionOp(matrix)
template<class binary_function>
__global__ static void CudaPerformReductionOnRowKernel(float* matrix, float* col,
  int m, int n, binary_function func)
{
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (row_id < m)
  {
    float r = matrix[row_id];
    for (int i = 0; i < n; i++)
    {
      r = func(r, matrix[i * m + row_id]);
    }
    col[row_id] = r;
    row_id += step;
  }
}

__global__ static void CudaPerformMaxIndexOnColKernel(float* matrix, float* row,
  int m, int n)
{
  // TODO: this is inefficient
  int col_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (col_id < n)
  {
    float maxv = matrix[col_id * m];
    int maxid = 0;
    for (int i = 0; i < m; i++)
    {
      if (matrix[col_id * m + i] > maxv)
      {
        maxv = matrix[col_id * m + i];
        maxid = i;
      }
    }
    row[col_id] = maxid;
    col_id += step;
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

//-----------------------------
// unary functions
class EXPOp
{
public:
  __device__ inline float operator()(float a) const
  {
    return expf(a);
  }
};

class LNOp
{
public:
  __device__ inline float operator()(float a) const
  {
    return logf(a);
  }
};

class NegativeOp
{
public:
  __device__ inline float operator()(float a) const
  {
    return -a;
  }
};

class SigmoidOp
{
public:
  __device__ inline float operator()(float a) const
  {
    return 1 / (1 + exp(-a));
  }
};

//--------------------------------
// binary function
class SubOp
{
public:
  __device__ inline float operator()(float a, float b) const
  {
    return a - b;
  }
};

class ReverseSubOp
{
public:
  __device__ inline float operator()(float a, float b) const
  {
    return b - a;
  }
};

class MultOp
{
public:
  __device__ inline float operator()(float a, float b) const
  {
    return a * b;
  }
};

class DivOp
{
public:
  __device__ inline float operator()(float a, float b) const
  {
    return a / b;
  }
};

class ReverseDivOp
{
public:
  __device__ inline float operator()(float a, float b) const
  {
    return b / a;
  }
};

class MaxOp
{
public:
  __device__ inline float operator()(float a, float b) const
  {
    return a > b ? a : b;
  }
};

class SumOp
{
public:
  __device__ inline float operator()(float a, float b) const
  {
    return a + b;
  }
};

//-------------------------------
// functions
void CudaPerformDotMult(float* a, float* b, float* c, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel <<<block, thread, 0, stream >> >(a, b, c, size, MultOp());
  CheckCudaError("CudaPerformDotMult");
}

void CudaPerformDotDiv(float* a, float* b, float* c, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel <<<block, thread, 0, stream >> >(a, b, c, size, DivOp());
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

void CudaPerformLeftConstSub(float* in, float* out, float val, size_t size, cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel <<<block, thread, 0, stream >> >(in, out, val, size, ReverseSubOp());
  CheckCudaError("CudaPerformLeftConstSub");
}

void CudaPerformLeftConstDiv(float* in, float* out, float val, size_t size, cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, val, size, ReverseDivOp());
  CheckCudaError("CudaPerformLeftConstDiv");
}

void CudaPerformNormAddOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel<<<block, thread, 0, stream >>>(matrix, row, res, m, n, SumOp());
  CheckCudaError("CudaPerformNormAddOnCol");
}

void CudaPerformNormSubOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel <<<block, thread, 0, stream >>>(matrix, row, res, m, n, SubOp());
  CheckCudaError("CudaPerformNormSubOnCol");
}

void CudaPerformNormMultOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel<<<block, thread, 0, stream >>>(matrix, row, res, m, n, MultOp());
  CheckCudaError("CudaPerformNormMultOnCol");
}

void CudaPerformNormDivOnCol(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnColKernel<<<block, thread, 0, stream >>>(matrix, row, res, m, n, DivOp());
  CheckCudaError("CudaPerformNormDivOnCol");
}

void CudaPerformNormAddOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel<<<block, thread, 0, stream >>>(matrix, row, res, m, n, SumOp());
  CheckCudaError("CudaPerformNormAddOnRow");
}

void CudaPerformNormSubOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel <<<block, thread, 0, stream >>>(matrix, row, res, m, n, SubOp());
  CheckCudaError("CudaPerformNormSubOnRow");
}

void CudaPerformNormMultOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel<<<block, thread, 0, stream >>>(matrix, row, res, m, n, MultOp());
  CheckCudaError("CudaPerformNormMultOnRow");
}

void CudaPerformNormDivOnRow(float* matrix, float* row, float* res, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformNormOnRowKernel<<<block, thread, 0, stream >>>(matrix, row, res, m, n, DivOp());
  CheckCudaError("CudaPerformNormDivOnRow");
}

void CudaPerformReductionSumOnCol(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformReductionOnColKernel<<<block, thread, 0, stream >>>(in, out, m, n, SumOp());
  CheckCudaError("CudaPerformReductionSumOnCol");
}

void CudaPerformReductionMaxOnCol(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformReductionOnColKernel<<<block, thread, 0, stream >>>(in, out, m, n, MaxOp());
  CheckCudaError("CudaPerformReductionMaxOnCol");
}

void CudaPerformReductionSumOnRow(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformReductionOnRowKernel<<<block, thread, 0, stream >>>(in, out, m, n, SumOp());
  CheckCudaError("CudaPerformReductionSumOnRow");
}

void CudaPerformReductionMaxOnRow(float* in, float* out, int m, int n, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformReductionOnRowKernel<<<block, thread, 0, stream >>>(in, out, m, n, MaxOp());
  CheckCudaError("CudaPerformReductionMaxOnRow");
}

void CudaPerformMaxIndexOnCol(float* in, float* out, int m, int n,
  cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(n, block, thread);
  CudaPerformMaxIndexOnColKernel <<<block, thread, 0, stream >>>(in, out, m, n);
  CheckCudaError("CudaPerformMaxIndexOnCol");
}

void CudaPerformMaxIndexOnRow(float* in, float* out, int m, int n,
  cudaStream_t stream)
{
  int block, thread;
  FindConfiguration(m, block, thread);
  CudaPerformMaxIndexOnColKernel << <block, thread, 0, stream >> >(in, out, m, n);
  CheckCudaError("CudaPerformMaxIndexOnRow");
}

void CudaPerformElewiseExp(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, size, EXPOp());
  CheckCudaError("CudaPerformEleWiseExp");
}

void CudaPerformElewiseLn(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, size, LNOp());
  CheckCudaError("CudaPerformEleWiseLn");
}

void CudaPerformElewiseSigmoid(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, size, SigmoidOp());
  CheckCudaError("CudaPerformEleWiseSigmoid");
}

void CudaPerformElewiseNegative(float* in, float* out, size_t size, cudaStream_t stream) {
  int block, thread;
  FindConfiguration(size, block, thread);
  CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, size, NegativeOp());
  CheckCudaError("CudaPerformEleWiseNegative");
}

}
}
