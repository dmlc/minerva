#include <cassert>
#include <functional>
#include <cstdio>
#include <cstdlib>

#include "chunk.h"
#include "dag/dag.h"
#include "op/physical_op.h"

using namespace std;

namespace minerva {

/*void MatrixMultiply(vector<DataNode*> inputs, vector<DataNode*> outputs) {
  cout << "Do matrix multiply" << endl;
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
  cout << "Do matrix addition" << endl;
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
  cout << "Do filling constants" << endl;
  float* a = DataStore::Instance().GetData(out->data_id(), DataStore::CPU);
  size_t len = out->meta().length;
  for(size_t i = 0; i < len; ++i)
    a[i] = val;
}*/

Chunk::Chunk(): data_node_(NULL) {
}
Chunk::Chunk(PhysicalDataNode* node): data_node_(node) {
}
Chunk::Chunk(const Chunk& other): data_node_(other.data_node_) {
}

std::vector<Chunk> Chunk::Compute(std::vector<Chunk> params,
    std::vector<Scale> result_sizes, PhysicalComputeFn* fn) {
  // TODO
  return std::vector<Chunk>();
}
Chunk Chunk::Generate(const Scale& result_size, PhysicalDataGenFn* fn) {
  // TODO
  return Chunk();
}

Chunk operator * (Chunk a, Chunk b) {
  // TODO
  return Chunk();
}

Chunk Chunk::Randn(const Scale& size, float mu, float var) {
  // TODO
  return Chunk();
}

Chunk Chunk::Constant(const Scale& size, float val) {
  // TODO
  return Chunk();
}

Chunk& Chunk::operator = (const Chunk& other) {
  data_node_ = other.data_node_;
  return *this;
}

/*void Chunk::Eval() {
  vector<uint64_t> targets{data_node_->node_id()};
  DagEngine::Instance().Process(Dag::Instance(), targets);
}

void Chunk::Print() {
  Eval();
  float *p = DataStore::Instance().GetData(data_node_->data_id(), DataStore::CPU);
  cout << "Size: " << data_node_->meta().size << endl;
  cout << "Data: [";
  size_t end = std::min(data_node_->meta().length, 10ul);
  for(size_t i = 0; i < end; ++i)
    cout << p[i] << " ";
  cout << "]" << endl;
}*/

Scale Chunk::Size() const {
  return data_node_->data_.size;
}
int Chunk::Size(int dim) const {
  return data_node_->data_.size[dim];
}
Chunk Chunk::Trans() {
  // TODO
  return Chunk();
}
Chunk Chunk::Merge(const NVector<Chunk>& partitions) {
  // TODO
  return Chunk();
}
NVector<Chunk> Chunk::Split(const Scale& numparts) {
  // TODO
  return NVector<Chunk>();
}

} // end of namespace minerva
