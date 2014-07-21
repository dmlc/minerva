#pragma once
#include "common/scale.h"
#include "common/nvector.h"
#include "dag/physical_dag.h"
#include "op/physical_op.h"
#include "op/physical_data.h"
#include "op/runner_wrapper.h"
#include <string>

namespace minerva {

class Chunk {
  friend class ChunkElewise;

 public:
  static Chunk Constant(const Scale&, float);
  static Chunk Randn(const Scale&, float, float);
  Chunk();
  Chunk(PhysicalDataNode* node);
  Chunk(const Chunk& other);
  PhysicalDataNode* data_node() const { return data_node_; }
  Chunk& operator=(const Chunk& other);

  // TODO to be implemented
  friend Chunk operator+(Chunk, Chunk);
  // friend Chunk operator - (Chunk , Chunk );
  // friend Chunk operator / (Chunk , Chunk );
  // friend Chunk operator + (Chunk , float );
  // friend Chunk operator - (Chunk , float );
  // friend Chunk operator * (Chunk , float );
  // friend Chunk operator / (Chunk , float );
  // friend Chunk operator + (float , Chunk );
  // friend Chunk operator - (float , Chunk );
  // friend Chunk operator * (float , Chunk );
  // friend Chunk operator / (float , Chunk );
  void operator += (Chunk);
  // void operator -= (Chunk );
  // void operator *= (Chunk );
  // void operator /= (Chunk );
  void operator += (float);
  // void operator -= (float );
  // void operator *= (float );
  // void operator /= (float );
  // void operator - ();
  friend Chunk operator*(Chunk, Chunk); // Matrix multiplication

  Scale Size() const;
  int Size(int) const;
  // TODO Functionality to split and merge
  Chunk Trans();
  static std::vector<Chunk> Compute(const std::vector<Chunk>&, const std::vector<Scale>& result_sizes, PhysicalComputeFn*);
  // TODO Possibly use shared_ptr
  static Chunk Generate(const Scale&, const std::string&, ClosureBase*);

 private:
  PhysicalDataNode* data_node_; // Set up in constructor
};

class ChunkElewise {
 public:
  // static Chunk Exp(Chunk);
  // static Chunk Ln(Chunk);
  // static Chunk Sigmoid(Chunk);
};

} // end of namespace minerva

