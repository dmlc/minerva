#pragma once
#include "common/scale.h"
#include "op/closure.h"
#include "dag/physical_dag.h"
#include "narray/io/file_format.h"
#include <initializer_list>

namespace minerva {

class NArray;
class IFileLoader;

class Elewise {
 public:
  static NArray Mult(NArray, NArray);
  static NArray Exp(NArray);
  static NArray Ln(NArray);
  static NArray Sigmoid(NArray);
};

class Convolution {
 public:
  static NArray ConvFF(NArray, NArray, const ConvInfo&);
  static NArray ConvBP(NArray, NArray, const ConvInfo&);
  static NArray GetGrad(NArray, NArray, const ConvInfo&);
};

class NArray {
  friend class Elewise;
  friend class Convolution;
  friend class MinervaSystem;
 public:
  static NArray Constant(const Scale& size, float val);
  static NArray Randn(const Scale& size, float mu, float var);
  static NArray LoadFromFile(const Scale& size, const std::string& fname, IFileLoader* loader,
      const Scale& numparts);
  static NArray Zeros(const Scale& size, const Scale& numparts) { return Constant(size, 0.0, numparts); }
  static NArray Ones(const Scale& size, const Scale& numparts) { return Constant(size, 1.0, numparts); }
  static NArray MakeNArray(const Scale&, std::shared_ptr<float>, const Scale&);
  // DAG generating operations
  static std::vector<NArray> Compute(std::vector<NArray> params,
      std::vector<Scale> result_sizes, LogicalComputeFn* fn);
  static NArray Generate(const Scale& size, LogicalDataGenFn* fn);


  NArray();
  NArray(const NArray& );
  NArray& operator = (const NArray& );
  ~NArray();
  friend NArray operator + (NArray, NArray);
  friend NArray operator - (NArray, NArray);
  friend NArray operator / (NArray, NArray);
  friend NArray operator + (float, NArray);
  friend NArray operator - (float, NArray);
  friend NArray operator * (float, NArray);
  friend NArray operator / (float, NArray);
  friend NArray operator + (NArray, float);
  friend NArray operator - (NArray, float);
  friend NArray operator * (NArray, float);
  friend NArray operator / (NArray, float);
  void operator += (NArray );
  void operator -= (NArray );
  void operator *= (NArray );
  void operator /= (NArray );
  void operator += (float );
  void operator -= (float );
  void operator *= (float );
  void operator /= (float );
  NArray operator - ();
  // matmult
  friend NArray operator * (NArray, NArray);
  // shape
  const Scale& Size() { return data_node_->data_.size; }
  int Size(int dim) { return data_node_->data_.size[dim]; }
  NArray Reshape(const Scale& dims);
  NArray Trans();
  // Lazy reductions
  NArray Sum(int dim);
  NArray Sum(const Scale& dims);
  NArray Max(int dim);
  NArray Max(const Scale& dims);
  NArray MaxIndex(int dim);
  // Replicate matrix
  NArray NormArithmetic(NArray, ArithmeticType);
  // Non-lazy reductions
  float Sum(); // TODO
  float Max(); // TODO
  int CountZero();


  // system
  void Eval();
  void EvalAsync();
  float* Get();
  void ToStream(std::ostream&, const FileFormat&);
  void ToFile(const std::string& filename, const FileFormat& );
  NArray RePartition(const NVector<Scale>& partitions);

 private:
  NArray(PhysicalDataNode* node);
  PhysicalDataNode* data_node_;
};

} // end of namespace minerva

