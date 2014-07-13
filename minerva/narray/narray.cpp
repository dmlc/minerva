#include "narray.h"
#include "op/logical_op.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

NArray::NArray(): data_node_(NULL) {}
NArray::NArray(LogicalDataNode* node): data_node_(node) {}

std::vector<NArray> NArray::Custom(std::vector<NArray> params, 
    std::vector<Scale> result_sizes, LogicalOp* op) {
  LogicalDag& ldag = MinervaSystem::Instance().logical_dag();
  std::vector<NArray> rst;
  std::vector<LogicalDataNode*> rst_data_nodes;
  for(Scale size : result_sizes) {
    LogicalDataNode* rst_node = ldag.NewDataNode(LogicalData{size});
    rst.push_back(NArray(rst_node));
    rst_data_nodes.push_back(rst_node);
  }
  std::vector<LogicalDataNode*> param_data_nodes;
  for(NArray p : params) {
    param_data_nodes.push_back(p.data_node_);
  }
  ldag.NewOpNode(param_data_nodes, rst_data_nodes, op);
  return rst;
}

NArray NArray::Constant(const Scale& size, float val, const Scale& parts) {
  // TODO
  return NArray();
}

NArray NArray::Randn(const Scale& size, float mu, float var, const Scale& parts) {
  RandnLogic* randn_op = new RandnLogic;
  randn_op->closure = {mu, var, parts};
  return NArray::Custom({}, {size}, randn_op)[0];
}

// matmult
NArray operator * (NArray lhs, NArray rhs) {
  // validity
  assert(lhs.Size().NumDims() == 2 && rhs.Size().NumDims() == 2);
  assert(lhs.Size(1) == rhs.Size(0));
  Scale newsize = {lhs.Size(0), rhs.Size(1)};
  MatMultLogic* matmult_op = new MatMultLogic;
  return NArray::Custom({lhs, rhs}, {newsize}, matmult_op)[0];
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
  TransLogic* trans_op = new TransLogic;
  return NArray::Custom({*this}, {newsize}, trans_op)[0];
}

} // end of namespace minerva
