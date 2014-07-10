#include "narray.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

NArray::NArray(): data_node_(NULL) {}
NArray::NArray(LogicalDataNode* node): data_node_(node) {}

NArray NArray::Constant(const Scale& size, float val) {
  // TODO
  return NArray();
}

NArray NArray::Randn(const Scale& size, float mu, float var) {
  // TODO
  return NArray();
}

// matmult
NArray operator * (NArray lhs, NArray rhs) {
  // validity
  assert(lhs.Size().NumDims() == 2 && rhs.Size().NumDims() == 2);
  assert(lhs.Size(1) == rhs.Size(0));
  // ldag construct
  Scale newsize = {lhs.Size(0), rhs.Size(1)};
  LogicalDag& ldag = MinervaSystem::Instance().logical_dag();
  LogicalDataNode* newnode = ldag.NewDataNode(LogicalData{newsize});
  ldag.NewOpNode(
      {lhs.data_node_, rhs.data_node_}, {newnode}, LogicalOp()
      );
  return NArray(newnode);
}

// shape
Scale NArray::Size() {
  return data_node_->data().size;
}

int NArray::Size(int dim) {
  return data_node_->data().size[dim];
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
  // TODO
  return NArray();
}

} // end of namespace minerva
