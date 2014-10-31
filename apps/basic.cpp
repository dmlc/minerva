#include <minerva.h>
#include <cublas_v2.h>

using namespace std;
using namespace minerva;

int main() {
  cublasHandle_t handle;
  void* a, * b, * c;
  int m = 3000, n = 2000;
  cublasCreate(&handle);
  cudaMalloc(&a, m * n * sizeof(float));
  cudaMalloc(&b, m * n * sizeof(float));
  cudaMalloc(&c, m * n * sizeof(float));
  float one = 1.0, minus_one = -1.0;
  cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, (float*) a, m, &minus_one, (float*) b, m, (float*) c, m);
  cudaFree(c);
  cudaFree(b);
  cudaFree(a);
  cublasDestroy(handle);
  return 0;
}
