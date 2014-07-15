#include "narray.h"
#include "op/logical_op.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

NArray::NArray(): data_node_(NULL) {}
NArray::NArray(LogicalDataNode* node): data_node_(node) {}

std::vector<NArray> NArray::Compute(std::vector<NArray> params, 
    std::vector<Scale> result_sizes, LogicalComputeFn* fn) {
  LogicalDag& ldag = MinervaSystem::Instance().logical_dag();
  std::vector<NArray> rst;
  std::vector<LogicalDataNode*> rst_data_nodes;
  for(Scale size : result_sizes) {
    LogicalDataNode* rst_node = ldag.NewDataNode({size, NULL});
    rst.push_back(NArray(rst_node));
    rst_data_nodes.push_back(rst_node);
  }
  std::vector<LogicalDataNode*> param_data_nodes;
  for(NArray p : params) {
    param_data_nodes.push_back(p.data_node_);
  }
  ldag.NewOpNode(param_data_nodes, rst_data_nodes, {fn});
  return rst;
}
  
NArray NArray::Generate(const Scale& size, LogicalDataGenFn* fn) {
  LogicalDag& ldag = MinervaSystem::Instance().logical_dag();
  LogicalDataNode* rst_node = ldag.NewDataNode({size, fn});
  return NArray(rst_node);
}

NArray NArray::Constant(const Scale& size, float val, const Scale& numparts) {
  FillOp* fill_op = new FillOp;
  fill_op->closure = {val, numparts};
  return NArray::Generate(size, fill_op);
}

NArray NArray::Randn(const Scale& size, float mu, float var, const Scale& numparts) {
  RandnOp* randn_op = new RandnOp;
  randn_op->closure = {mu, var, numparts};
  return NArray::Generate(size, randn_op);
}

// matmult
NArray operator * (NArray lhs, NArray rhs) {
  // validity
  assert(lhs.Size().NumDims() == 2 && rhs.Size().NumDims() == 2);
  assert(lhs.Size(1) == rhs.Size(0));
  Scale newsize = {lhs.Size(0), rhs.Size(1)};
  MatMultOp* matmult_op = new MatMultOp;
  return NArray::Compute({lhs, rhs}, {newsize}, matmult_op)[0];
}

// shape
Scale NArray::Size() {
  return data_node_->data_.size;
}

int NArray::Size(int dim) {
  return data_node_->data_.size[dim];
}

NArray NArray::Tile(const Scale& times) {
  // TODO
  return NArray();
}

NArray NArray::Reshape(const Scale& dims) {
  // TODO
  return NArray();
}

NArray NArray::Trans() {
  // validity
  assert(Size().NumDims() == 2);
  Scale newsize = {Size(1), Size(0)};
  TransOp* trans_op = new TransOp;
  return NArray::Compute({*this}, {newsize}, trans_op)[0];
}

} // end of namespace minerva
