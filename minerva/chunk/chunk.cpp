#include "chunk.h"
#include "dag/dag.h"
#include "op/physical_op.h"
#include "system/minerva_system.h"
#include <cassert>
#include <functional>
#include <cstdio>
#include <cstdlib>

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

Chunk operator+(const Chunk& a, const Chunk& b) {
  assert(a.Size() == b.Size());
  ArithmeticClosure closure{ADD};
  return Chunk::Compute({a, b}, {a.Size()}, "arithmetic", NewClosureBase(closure))[0];
}

Chunk operator+(const Chunk& a, float f) {
  ArithmeticConstClosure closure{ADD, f, 1};
  return Chunk::Compute({a}, {a.Size()}, "arithmeticConstant", NewClosureBase(closure))[0];
}

Chunk operator+(float f, const Chunk& a) {
  ArithmeticConstClosure closure{ADD, f, 0};
  return Chunk::Compute({a}, {a.Size()}, "arithmeticConstant", NewClosureBase(closure))[0];
}

void Chunk::operator+=(const Chunk& o) {
  *this = *this + o;
}

void Chunk::operator+=(float f) {
  *this = *this + f;
}

Chunk Chunk::Constant(const Scale& size, float val) {
  FillClosure closure{val};
  return Chunk::Generate(size, "fill", NewClosureBase(closure));
}

Chunk Chunk::Randn(const Scale& size, float mu, float var) {
  // RandnOp* randn_op = new RandnOp;
  // randn_op->closure = {mu, var};
  // return Chunk::Generate(size, randn_op);
  return Chunk();
}

Chunk::Chunk(): data_node_(NULL) {
}

Chunk::Chunk(PhysicalDataNode* node): data_node_(node) {
}

Chunk::Chunk(const Chunk& other): data_node_(other.data_node_) {
}

Chunk& Chunk::operator=(const Chunk& other) {
  if (this == &other) {
    return *this;
  }
  data_node_ = other.data_node_;
  return *this;
}

Chunk operator*(const Chunk& a, const Chunk& b) {
  assert(a.Size().NumDims() == 2 && b.Size().NumDims() == 2); // 2D multiplication
  assert(a.Size(1) == b.Size(0));
  Scale new_size{a.Size(0), b.Size(1)};
  MatMultClosure closure;
  return Chunk::Compute({a, b}, {new_size}, "matMult", NewClosureBase(closure))[0];
}

Scale Chunk::Size() const {
  return data_node_->data_.size;
}

int Chunk::Size(int dim) const {
  return data_node_->data_.size[dim];
}

Chunk Chunk::Trans() {
  assert(Size().NumDims() == 2); // 2D transposing
  Scale new_size{Size(1), Size(0)};
  TransposeClosure closure;
  return Chunk::Compute({*this}, {new_size}, "trans", NewClosureBase(closure))[0];
}

vector<Chunk> Chunk::Compute(const vector<Chunk>& params, const vector<Scale>& result_sizes, const string& runner_name, ClosureBase* closure) {
  auto& pdag = MinervaSystem::Instance().physical_dag();
  auto& pengine = MinervaSystem::Instance().physical_engine();
  vector<Chunk> rst;
  vector<PhysicalDataNode*> rst_data_nodes;
  for (auto& size: result_sizes) {
    auto rst_node = pdag.NewDataNode(PhysicalData(size));
    rst.push_back(Chunk(rst_node));
    rst_data_nodes.push_back(rst_node);
  }
  vector<PhysicalDataNode*> param_data_nodes;
  for (auto& ch: params) {
    param_data_nodes.push_back(ch.data_node());
  }
  PhysicalOp op;
  op.runner_id = pengine.GetRunnerID(runner_name);
  op.closure = closure;
  pdag.NewOpNode(param_data_nodes, rst_data_nodes, op);
  return rst;
}

Chunk Chunk::Generate(const Scale& result_size, const string& runner_name, ClosureBase* closure) {
  auto& pdag = MinervaSystem::Instance().physical_dag();
  auto& pengine = MinervaSystem::Instance().physical_engine();
  PhysicalData pdata(result_size);
  pdata.generator_id = pengine.GetRunnerID(runner_name);
  pdata.closure = closure;
  PhysicalDataNode* rst_node = pdag.NewDataNode(pdata);
  return Chunk(rst_node);
}

} // end of namespace minerva

