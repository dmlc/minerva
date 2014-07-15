#pragma once

#include "common/scale.h"
#include "common/nvector.h"
#include "dag/physical_dag.h"

namespace minerva {

class Chunk;
class ChunkElewise;

class ChunkElewise {
 public:
  Chunk Mult(Chunk, Chunk);
  Chunk Exp(Chunk );
  Chunk Ln(Chunk );
  Chunk Sigmoid(Chunk );
};

class Chunk {
  friend class ChunkElewise;
 public:
  static Chunk Constant(const Scale& size, float val);
  static Chunk Randn(const Scale& size, float mu, float var);

 public:
  Chunk();
  Chunk(PhysicalDataNode* node);
  Chunk(const Chunk& other);
  PhysicalDataNode* data_node() const { return data_node_; }
  Chunk& operator = (const Chunk& other);

  // elewise
  friend Chunk operator + (Chunk , Chunk );
  friend Chunk operator - (Chunk , Chunk );
  friend Chunk operator / (Chunk , Chunk );
  friend Chunk operator + (Chunk , float );
  friend Chunk operator - (Chunk , float );
  friend Chunk operator * (Chunk , float );
  friend Chunk operator / (Chunk , float );
  friend Chunk operator + (float , Chunk );
  friend Chunk operator - (float , Chunk );
  friend Chunk operator * (float , Chunk );
  friend Chunk operator / (float , Chunk );

  void operator += (Chunk );
  void operator -= (Chunk );
  void operator *= (Chunk );
  void operator /= (Chunk );
  void operator += (float );
  void operator -= (float );
  void operator *= (float );
  void operator /= (float );
  void operator - ();

  // matmult
  friend Chunk operator * (Chunk a, Chunk b);

  // reduction
  // TODO

  // shape
  Scale Size() const;
  int Size(int dim) const;
  Chunk Trans();
  static Chunk Merge(const NVector<Chunk>& );
  NVector<Chunk> Split(const Scale& numparts);

  // customized operations
  static std::vector<Chunk> Compute(std::vector<Chunk> params,
      std::vector<Scale> result_sizes, PhysicalComputeFn* fn);
  static Chunk Generate(const Scale& result_size, PhysicalDataGenFn* fn);

 private:
  PhysicalDataNode* data_node_; // Set up in constructor
};

} // end of namespace minerva
