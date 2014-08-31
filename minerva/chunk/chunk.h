#pragma once
#include "common/scale.h"
#include "common/nvector.h"
#include "dag/physical_dag.h"
#include "op/closure.h"
#include "device/device_info.h"

namespace minerva {

class Chunk {
 public:
  Chunk();
  Chunk(PhysicalDataNode* node);
  Chunk(const Chunk& other);
  PhysicalDataNode* data_node() const { return data_node_; }
  Chunk& operator=(const Chunk& other);
  // Shape
  const Scale& Size() const { return data_node_->data_.size; }
  int Size(int dim) const { return data_node_->data_.size[dim]; }

  // TODO deprecated, no partition for now
  static Scale ComputeOffset(NVector<Chunk>); // Return merged size
  static Chunk Merge(NVector<Chunk>);
  NVector<Chunk> Split(const NVector<Scale>& partsizes);
  // DAG building operations
  static std::vector<Chunk> Compute(const std::vector<Chunk>& params,
      const std::vector<Scale>& result_sizes, PhysicalComputeFn* fn);

 private:
  PhysicalDataNode* data_node_; // Set up in constructor
};

} // end of namespace minerva

