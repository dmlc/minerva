#include "op/impl/cuda/cuda_perform.h"
#include "op/op_types.h"
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
    float* mat = matrix + row_id;
    for (int i = 0; i < n; i++)
    {
      res[i*n + row_id] = func(mat[i*m + row_id], row[i]);
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
    float* mat = matrix + row_id;
    for (int i = 0; i < n; i++)
    {
      res[i*n + row_id] = func(mat[i*m + row_id], col[row_id]);
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
    float* mat = matrix + row_id;
    float r = *mat;
    for (int i = 0; i < m; i++)
    {
      r = func(r, mat[col_id*n + i]);
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
    float* mat = matrix + row_id;
    float r = *mat;
    for (int i = 0; i < n; i++)
    {
      r = func(r, mat[i*m + row_id]);
    }
    col[row_id] = r;
    row_id += step;
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
#if 0
      return 1 / (1 + exp(-a));
#else
      return a / (1 + abs(a));
#endif
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

  void CudaPerformNormOnCol(float* matrix, float* row, float* res, int m, int n,
    const ArithmeticType & type, cudaStream_t stream)
  {
    int block, thread;
    FindConfiguration(m, block, thread);
    switch (type)
    {
    case ADD:
      CudaPerformNormOnColKernel <<<block, thread, 0, stream >>>(matrix, row, res, m, n, SumOp());
      break;
    case SUB:
      CudaPerformNormOnColKernel <<<block, thread, 0, stream >>>(matrix, row, res, m, n, SubOp());
      break;
    case MULT:
      CudaPerformNormOnColKernel <<<block, thread, 0, stream >>>(matrix, row, res, m, n, MultOp());
      break;
    case DIV:
      CudaPerformNormOnColKernel <<<block, thread, 0, stream >>>(matrix, row, res, m, n, DivOp());
      break;
    }    
    CheckCudaError("CudaPerformNormOnCol");
  }

  void CudaPerformNormOnRow(float* matrix, float* col, float* res, int m, int n,
    const ArithmeticType & type, cudaStream_t stream)
  {
    int block, thread;
    FindConfiguration(m, block, thread);
    switch (type)
    {
    case ADD:
      CudaPerformNormOnRowKernel <<<block, thread, 0, stream >>>(matrix, col, res, m, n, SumOp());
      break;
    case SUB:
      CudaPerformNormOnRowKernel <<<block, thread, 0, stream >>>(matrix, col, res, m, n, SubOp());
      break;
    case MULT:
      CudaPerformNormOnRowKernel <<<block, thread, 0, stream >>>(matrix, col, res, m, n, MultOp());
      break;
    case DIV:
      CudaPerformNormOnRowKernel <<<block, thread, 0, stream >>>(matrix, col, res, m, n, DivOp());
      break;
    }
    CheckCudaError("CudaPerformNormOnRow");
  }

  void CudaPerformEleWise(float* in, float* out, size_t size, 
    const ElewiseType & type, cudaStream_t stream)
  {
    int block, thread;
    FindConfiguration(size, block, thread);
    switch (type)
    {
    case EXP:
      CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, 0, size, EXPOp());
      break;
    case LN:
      CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, 0, size, LNOp());
      break;
    case SIGMOID:
      CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, 0, size, SigmoidOp());
      break;
    case NEGATIVE:
      CudaPerformDotKernel <<<block, thread, 0, stream >>>(in, out, 0, size, NegativeOp());
      break;
    }  
    CheckCudaError("CudaPerformEleWise");
  }

  void CudaPerformReductionOnCol(float* in, float* out, int m, int n,
    const ReductionType & type, cudaStream_t stream)
  {
    int block, thread;
    FindConfiguration(n, block, thread);
    switch (type)
    {
    case SUM:
      CudaPerformReductionOnColKernel <<<block, thread, 0, stream >>>(in, out, m, n, SumOp());
      break;
    case MAX:
      CudaPerformReductionOnColKernel <<<block, thread, 0, stream >>>(in, out, m, n, MaxOp);
      break;
    }    
    CheckCudaError("CudaPerformLeftConstDiv");
  }

  void CudaPerformReductionOnRow(float* in, float* out, int m, int n,
    const ReductionType & type, cudaStream_t stream)
  {
    int block, thread;
    FindConfiguration(m, block, thread);
    switch (type)
    {
    case SUM:
      CudaPerformReductionOnRowKernel <<<block, thread, 0, stream >>>(in, out, m, n, SumOp());
      break;
    case MAX:
      CudaPerformReductionOnRowKernel <<<block, thread, 0, stream >>>(in, out, m, n, MaxOp);
      break;
    }    
    CheckCudaError("CudaPerformLeftConstDiv");
  }



}
}
