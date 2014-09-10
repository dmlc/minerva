#pragma once
#include "common/scale.h"
#include "op/closure.h"
#include "dag/physical_dag.h"
#include "narray/io/file_format.h"
#include "narray/io/file_loader.h"
#include <initializer_list>
#include <memory>

namespace minerva {

class NArray {
  friend class Elewise;
  friend class Convolution;
  friend class MinervaSystem;
 public:
  // Static constructors
  static NArray Constant(const Scale& size, float val);
  static NArray Randn(const Scale& size, float mu, float var);
  static NArray LoadFromFile(const Scale& size, const std::string& fname, IFileLoader* loader);
  static NArray Zeros(const Scale& size);
  static NArray Ones(const Scale& size);
  static NArray MakeNArray(const Scale& size, std::shared_ptr<float> array);
  // DAG generating operations
  static std::vector<NArray> Compute(
      const std::vector<NArray>& params,
      const std::vector<Scale>& result_sizes,
      LogicalComputeFn* fn);
  static NArray ComputeOne(
      const std::vector<NArray>& params,
      const Scale& size,
      PhysicalComputeFn* fn);
  static NArray GenerateOne(
      const Scale& size,
      PhysicalComputeFn* fn);
  // Constructors and destructors
  NArray();
  NArray(const NArray&);
  NArray& operator=(const NArray&);
  ~NArray();
  // Element-wise operations
  friend NArray operator+(NArray, NArray);
  friend NArray operator-(NArray, NArray);
  friend NArray operator/(NArray, NArray);
  friend NArray operator+(float, NArray);
  friend NArray operator-(float, NArray);
  friend NArray operator*(float, NArray);
  friend NArray operator/(float, NArray);
  friend NArray operator+(NArray, float);
  friend NArray operator-(NArray, float);
  friend NArray operator*(NArray, float);
  friend NArray operator/(NArray, float);
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
  NArray(PhysicalDataNode*);
  PhysicalDataNode* data_node_;
};

} // end of namespace minerva

