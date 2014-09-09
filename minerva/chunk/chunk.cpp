#include "chunk.h"
#include "dag/physical_dag.h"
#include "op/physical_op.h"
#include "op/shared_op.h"
#include "system/minerva_system.h"
#include "system/data_store.h"
#include "common/common.h"
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <glog/logging.h>

using namespace std;

namespace minerva {

Chunk::Chunk(): data_node_(NULL) { }
Chunk::Chunk(PhysicalDataNode* node): data_node_(node) { }
Chunk::Chunk(const Chunk& other): data_node_(other.data_node_) { }
Chunk& Chunk::operator = (const Chunk& other) {
  data_node_ = other.data_node_;
  return *this;
}

Scale Chunk::ComputeOffset(NVector<Chunk> chunks) {
  const Scale& numparts = chunks.Size();
  size_t numdims = numparts.NumDims();
  Scale pos = Scale::Origin(numdims);
  while(1) {
    auto& phy_data = chunks[pos].data_node()->data_;
    phy_data.offset.Resize(numdims, 0);
    phy_data.offset_index = pos;
    Scale upleftpos = pos.Map([] (int x) { return max(x - 1, 0); });
    auto& upleft_phy_data = chunks[upleftpos].data_node()->data_;
    for(size_t i = 0; i < numdims; ++i) {
      if(pos[i] != 0) { // if the index of this dimension is 0, the offset should be 0 too
        phy_data.offset[i] = upleft_phy_data.offset[i] + upleft_phy_data.size[i];
      }
    }
    if(!Scale::IncrOne(pos, numparts)) {
      return phy_data.offset + phy_data.size;
    }
  }
}

Chunk Chunk::Merge(NVector<Chunk> parts) {
  Scale rstsize = ComputeOffset(parts);
  AssembleOp* assemble_op = new AssembleOp;
  return Chunk::Compute(parts.ToVector(), {rstsize}, assemble_op)[0];
}

NVector<Chunk> Chunk::Split(const NVector<Scale>& partsizes) {
  SplitOp* split_op = new SplitOp;
  const auto& rstchunklst = Compute({*this}, partsizes.ToVector(), split_op);
  NVector<Chunk> rstchunknvec(rstchunklst, partsizes.Size());
  ComputeOffset(rstchunknvec);
  return rstchunknvec;
}

// DAG building operations
vector<Chunk> Chunk::Compute(
      const vector<Chunk>& params,
      const vector<Scale>& result_sizes,
      PhysicalComputeFn* fn) {
  auto& ms = MinervaSystem::Instance();
  auto& pdag = ms.physical_dag();
  auto& device_info = fn->device_info;
  auto rst = Map<Chunk>(result_sizes, [&](const Scale& size) {
    PhysicalData phy_data;
    phy_data.size = size;
    phy_data.data_id = ms.data_store().GenerateDataId();
    phy_data.device_info = device_info;
    auto rst_node = pdag.NewDataNode(phy_data);
    return Chunk(rst_node);
  });
  auto rst_data_nodes = Map<PhysicalDataNode*>(rst, [](const Chunk& i) {
    return i.data_node();
  });
  auto param_data_nodes = Map<PhysicalDataNode*>(params, [](const Chunk& i) {
    return i.data_node();
  });
  PhysicalOp phy_op;
  phy_op.impl_type = ImplType::kNA;
  phy_op.compute_fn = fn;
  pdag.NewOpNode(param_data_nodes, rst_data_nodes, phy_op);
  return rst;
}

} // end of namespace minerva

