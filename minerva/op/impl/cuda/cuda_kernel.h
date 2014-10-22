#pragma once

// Unary functions
class ExpOp {
 public:
  __device__ inline float operator()(float a) const {
    return expf(a);
  }
};

class LnOp {
 public:
  __device__ inline float operator()(float a) const {
    return logf(a);
  }
};

class NegativeOp {
 public:
  __device__ inline float operator()(float a) const {
    return -a;
  }
};

class SigmoidOp {
 public:
  __device__ inline float operator()(float a) const {
    return 1 / (1 + exp(-a));
  }
};

// Binary function
class SubOp {
 public:
  __device__ inline float operator()(float a, float b) const {
    return a - b;
  }
};

class ReverseSubOp {
 public:
  __device__ inline float operator()(float a, float b) const {
    return b - a;
  }
};

class MultOp {
 public:
  __device__ inline float operator()(float a, float b) const {
    return a * b;
  }
};

class DivOp {
 public:
  __device__ inline float operator()(float a, float b) const {
    return a / b;
  }
};

class ReverseDivOp {
 public:
  __device__ inline float operator()(float a, float b) const {
    return b / a;
  }
};

class MaxOp {
 public:
  __device__ inline float operator()(float a, float b) const {
    return b < a ? a : b;
  }
};

class SumOp {
 public:
  __device__ inline float operator()(float a, float b) const {
    return a + b;
  }
};

// c = a DotOp b
template<typename Func>
__global__ static void CudaPerformDotKernel(float* a, float* b, float* c, size_t size, Func func) {
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    *(c + cur) = func(*(a + cur), *(b + cur));
    cur += gridDim.x * blockDim.x;
  }
}

// y = x Op v
template<typename Func>
__global__ static void CudaPerformDotKernel(float* x, float* y, float v, size_t size, Func func) {
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    y[cur] = func(x[cur], v);
    cur += gridDim.x * blockDim.x;
  }
}

// y = Op (x)
template<typename Func>
__global__ static void CudaPerformDotKernel(float* x, float* y, size_t size, Func func) {
  int cur = threadIdx.x + blockIdx.x * blockDim.x;
  while (cur < size) {
    y[cur] = func(x[cur]);
    cur += gridDim.x * blockDim.x;
  }
}

// res = matrix Norm row
template<typename Func>
__global__ static void CudaPerformNormOnColKernel(float* matrix, float* row, float * res, int m, int n, Func func) {
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (row_id < m) {
    for (int i = 0; i < n; ++i) {
      res[i*m + row_id] = func(matrix[i*m + row_id], row[i]);
    }
    row_id += step;
  }
}

// res = matrix Norm col
template<typename Func>
__global__ static void CudaPerformNormOnRowKernel(float* matrix, float* col, float * res, int m, int n, Func func) {
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (row_id < m) {
    for (int i = 0; i < n; ++i) {
      res[i*m + row_id] = func(matrix[i*m + row_id], col[row_id]);
    }
    row_id += step;
  }
}

// row = ReductionOp(matrix)
template<typename Func>
__global__ static void CudaPerformReductionOnColKernel(float* matrix, float* row, int m, int n, Func func) {
  // TODO: this is inefficient
  int col_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (col_id < n) {
    float r = matrix[col_id * m];
    for (int i = 0; i < m; ++i) {
      r = func(r, matrix[col_id * m + i]);
    }
    row[col_id] = r;
    col_id += step;
  }
}

// col = ReductionOp(matrix)
template<typename Func>
__global__ static void CudaPerformReductionOnRowKernel(float* matrix, float* col, int m, int n, Func func) {
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (row_id < m) {
    float r = matrix[row_id];
    for (int i = 0; i < n; ++i) {
      r = func(r, matrix[i * m + row_id]);
    }
    col[row_id] = r;
    row_id += step;
  }
}

// row = MaxIndexOp(matrix)
__global__ static void CudaPerformMaxIndexOnColKernel(float* matrix, float* row, int m, int n) {
  // TODO: this is inefficient
  int col_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (col_id < n) {
    float maxv = matrix[col_id * m];
    int maxid = 0;
    for (int i = 0; i < m; ++i) {
      if (matrix[col_id * m + i] > maxv) {
        maxv = matrix[col_id * m + i];
        maxid = i;
      }
    }
    row[col_id] = maxid;
    col_id += step;
  }
}

// col = MaxIndexOp(matrix)
__global__ static void CudaPerformMaxIndexOnRowKernel(float* matrix, float* col, int m, int n) {
  int row_id = threadIdx.x + blockIdx.x * blockDim.x;
  int step = gridDim.x * blockDim.x;
  while (row_id < m) {
    float maxv = matrix[row_id];
    int maxid = 0;
    for (int i = 0; i < n; ++i) {
      if (matrix[i * m + row_id] > maxv) {
        maxv = matrix[i * m + row_id];
        maxid = i;
      }
    }
    col[row_id] = maxid;
    row_id += step;
  }
}

