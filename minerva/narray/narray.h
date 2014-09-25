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
  static NArray LoadFromFile(const Scale& size, const std::string& fname, std::shared_ptr<IFileLoader> loader);
  static NArray Zeros(const Scale& size);
  static NArray Ones(const Scale& size);
  static NArray MakeNArray(const Scale& size, std::shared_ptr<float> array);
  // DAG generating operations
  static std::vector<NArray> Compute(
      const std::vector<NArray>& params,
      const std::vector<Scale>& result_sizes,
      PhysicalComputeFn* fn);
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
  NArray(NArray&&);
  NArray& operator=(const NArray&);
  NArray& operator=(NArray&&);
  ~NArray();
  // Element-wise operations
  friend NArray operator+(const NArray&, const NArray&);
  friend NArray operator-(const NArray&, const NArray&);
  friend NArray operator/(const NArray&, const NArray&);
  friend NArray operator+(float, const NArray&);
  friend NArray operator-(float, const NArray&);
  friend NArray operator*(float, const NArray&);
  friend NArray operator/(float, const NArray&);
  friend NArray operator+(const NArray&, float);
  friend NArray operator-(const NArray&, float);
  friend NArray operator*(const NArray&, float);
  friend NArray operator/(const NArray&, float);
  void operator+=(const NArray&);
  void operator-=(const NArray&);
  void operator*=(const NArray&);
  void operator/=(const NArray&);
  void operator+=(float);
  void operator-=(float);
  void operator*=(float);
  void operator/=(float);
  NArray operator-();
  // Matmult
  friend NArray operator*(const NArray&, const NArray&);
  // Shape
  const Scale& Size() const { return data_node_->data_.size; }
  int Size(int dim) const { return data_node_->data_.size[dim]; }
  NArray Reshape(const Scale& dims) const;  // TODO
  NArray Trans() const;
  // Lazy reductions
  NArray Sum(int dim) const;
  NArray Sum(const Scale& dims) const;
  NArray Max(int dim) const;
  NArray Max(const Scale& dims) const;
  NArray MaxIndex(int dim) const;
  // Replicate matrix
  NArray NormArithmetic(const NArray&, ArithmeticType) const;
  // Non-lazy reductions
  float Sum() const;  // TODO
  float Max() const;  // TODO
  int CountZero() const;
  // System
  void Eval() const;
  void EvalAsync() const;
  std::shared_ptr<float> Get() const;
  void ToStream(std::ostream& out, const FileFormat& format) const;
  void ToFile(const std::string& filename, const FileFormat& format) const;

 private:
  NArray(PhysicalDataNode*);
  PhysicalDataNode* data_node_;
};

// Matmult
NArray operator*(const NArray&, const NArray&);

} // end of namespace minerva

