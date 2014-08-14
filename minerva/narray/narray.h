#pragma once

#include <cstdlib>
#include <vector>
#include <initializer_list>
#include <ostream>

#include "common/scale.h"
#include "op/closure.h"
#include "dag/logical_dag.h"

namespace minerva {

class NArray;
class Elewise;
class Convolution;
struct FileFormat;

class MinervaSystem;
class IFileLoader;

class Elewise {
 public:
  static NArray Mult(NArray, NArray);
  static NArray Exp(NArray );
  static NArray Ln(NArray );
  static NArray Sigmoid(NArray );
};

class Convolution {
 public:
  static NArray ConvFF(NArray act, NArray weight, const ConvInfo& convinfo);
  static NArray ConvBP(NArray sen, NArray weight, const ConvInfo& convinfo);
  static NArray GetGrad(NArray act_low, NArray sen_high, const ConvInfo& convinfo);
};

struct FileFormat {
  bool binary; // whether output in binary
};

class NArray {
  friend class Elewise;
  friend class Convolution;
  friend class MinervaSystem;
 public:
  static NArray Constant(const Scale& size, float val,
      const NVector<Scale>&);
  static NArray Randn(const Scale& size, float mu, float var,
      const NVector<Scale>&);
  static NArray Constant(const Scale& size, float val, const Scale& );
  static NArray Randn(const Scale& size, float mu, float var, const Scale& );
  static NArray LoadFromFile(const Scale& size, const std::string& fname, IFileLoader* loader,
      const Scale& numparts);
  static NArray LoadFromArray(const Scale&, float*, const Scale&);
  NArray();
  NArray(const NArray& );
  NArray& operator = (const NArray& );
  ~NArray();
 public:
  // element-wise
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

  // customize operator
  static std::vector<NArray> Compute(std::vector<NArray> params,
      std::vector<Scale> result_sizes, LogicalComputeFn* fn);
  static NArray Generate(const Scale& size, LogicalDataGenFn* fn, const NVector<Scale>& parts);
  static NArray Generate(const Scale& size, LogicalDataGenFn* fn, const Scale& numparts);

  // system
  void Eval();
  float* Get();
  void ToStream(std::ostream&, const FileFormat&);
  void ToFile(const std::string& filename, const FileFormat& );
  NArray RePartition(const NVector<Scale>& partitions);

 private:
  NArray(LogicalDataNode* node);
  LogicalDataNode* data_node_;
};

} // end of namespace minerva
