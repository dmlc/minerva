#include <cassert>
#include <functional>
#include <cstdio>

#include "chunk.h"
#include "dag/dag.h"
#include "procedures/dag_engine.h"
#include "system/data_store.h"

using namespace std;

namespace minerva {

void MatrixMultiply(vector<DataNode*> inputs, vector<DataNode*> outputs) {
  float* a = DataStore::Instance().GetData(inputs[0]->data_id(), DataStore::CPU);
  float* b = DataStore::Instance().GetData(inputs[1]->data_id(), DataStore::CPU);
  float* c = DataStore::Instance().GetData(outputs[0]->data_id(), DataStore::CPU);
  int m = outputs[0]->meta().size[0];
  int n = outputs[0]->meta().size[1];
  int k = inputs[0]->meta().size[1];
  for(int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j) {
      c[i * n + j] = 0.0;
      for(int l = 0; l < k; ++l)
        c[i * n + j] += a[i * k + l] * b[l * n + j];
    }
}
void MatrixAdd(vector<DataNode*> inputs, DataNode* output) {
  vector<float*> inptrs;
  for(auto innode : inputs)
    inptrs.push_back(DataStore::Instance().GetData(innode->data_id(), DataStore::CPU));
  float* outptrs = DataStore::Instance().GetData(output->data_id(), DataStore::CPU);
  size_t len = output->meta().length;
  for(size_t i = 0; i < len; ++i) {
    outptrs[i] = 0.0;
    for(size_t j = 0; j < inptrs.size(); ++j)
      outptrs[i] += inptrs[j][i];
  }
}
void FillConstant(DataNode* out, float val) {
  float* a = DataStore::Instance().GetData(out->data_id(), DataStore::CPU);
  size_t len = out->meta().length;
  for(size_t i = 0; i < len; ++i)
    a[i] = val;
}

Chunk::Chunk(): data_node_(NULL) {
}
Chunk::Chunk(const Index& size) {
  data_node_ = Dag::Instance().NewDataNode(DataNodeMeta(size));
}

Chunk operator*(const Chunk& a, const Chunk& b) {
  Index asize = a.Size(), bsize = b.Size();
  // Check if operands match in dimension.
  assert(asize[1] == bsize[0]);
  Index retsize = {asize[0], bsize[1]};
  Chunk ret(retsize);
  vector<DataNode*> in_nodes = {a.data_node(), b.data_node()};
  vector<DataNode*> out_nodes = {ret.data_node()};
  OpNode::Runner multrunner = bind(&MatrixMultiply, in_nodes, out_nodes);
  Dag::Instance().NewOpNode(
      {a.data_node(), b.data_node()}, {ret.data_node()},
      multrunner, OpNodeContext());
  return ret;
}

Chunk operator + (const Chunk& a, const Chunk& b) {
  Index asize = a.Size(), bsize = b.Size();
  // checking
  assert(asize == bsize);
  Chunk ret(asize); 
  vector<DataNode*> in_nodes = {a.data_node(), b.data_node()};
  OpNode::Runner addrunner = bind(&MatrixAdd, in_nodes, ret.data_node());
  Dag::Instance().NewOpNode({a.data_node(), b.data_node()}, {ret.data_node()},
      addrunner, OpNodeContext());
  return ret;
}

Chunk Chunk::Constant(const Index& size, float val) {
  Chunk ret(size);
  OpNode::Runner fillrunner = bind(&FillConstant, ret.data_node(), val);
  Dag::Instance().NewOpNode({}, {ret.data_node()}, fillrunner, OpNodeContext());
  return ret;
}

void Chunk::operator += (const Chunk& a) {
  *this = (*this) + a;
}

void Chunk::Eval() {
  vector<uint64_t> targets{data_node_->node_id()};
  DagEngine::Instance().Process(Dag::Instance(), targets);
}

} // end of namespace minerva
