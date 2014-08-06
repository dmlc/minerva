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
      const NVector<PartInfo>& = NVector<PartInfo>());
  static NArray Randn(const Scale& size, float mu, float var, 
      const NVector<PartInfo>& = NVector<PartInfo>());
  static NArray Constant(const Scale& size, float val, const Scale& );
  static NArray Randn(const Scale& size, float mu, float var, const Scale& );
  static NArray LoadFromFile(const Scale& size, const std::string& fname, IFileLoader* loader,
      const Scale& numparts);
  static NArray LoadFromArray(const Scale&, float*, const Scale&);
  NArray();
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
  // lazy reductions
  NArray Sum(int dim);
  NArray Sum(const Scale& dims);
  NArray Max(int dim);
  NArray Max(const Scale& dims);
  NArray MaxIndex(int dim);
  NArray MaxIndex(const Scale& dims);
  // non-lazy reductions
  float Sum();
  float Max();
  int CountZero();
  // shape
  Scale Size();
  int Size(int dim);
  NArray Tile(const Scale& times);
  NArray Reshape(const Scale& dims);
  NArray Trans();

  // customize operator
  static std::vector<NArray> Compute(std::vector<NArray> params,
      std::vector<Scale> result_sizes, LogicalComputeFn* fn);
  static NArray Generate(const Scale& size, LogicalDataGenFn* fn, const NVector<PartInfo>& parts);
  static NArray Generate(const Scale& size, LogicalDataGenFn* fn, const Scale& numparts);

  // system
  void Eval();
  float* Get();
  void ToStream(std::ostream&, const FileFormat&);
  void ToFile(const std::string& filename, const FileFormat& );
  NArray RePartition(const NVector<PartInfo>& partitions);

 private:
  NArray(LogicalDataNode* node);
  LogicalDataNode* data_node_;
};

} // end of namespace minerva
