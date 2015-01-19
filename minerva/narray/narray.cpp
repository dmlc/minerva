#include "narray/narray.h"
#include "op/physical_op.h"
#include "common/common.h"
#include "system/minerva_system.h"
#include "io/file_loader.h"
#include <fstream>
#include <glog/logging.h>
#include <iomanip>

using namespace std;

namespace minerva {

// Static constructors
NArray NArray::Constant(const Scale& size, float val) {
  FillOp* fill_op = new FillOp();
  fill_op->closure = {val};
  return NArray::GenerateOne(size, fill_op);
}

NArray NArray::Randn(const Scale& size, float mu, float var) {
  RandnOp* randn_op = new RandnOp();
  randn_op->closure = {mu, var};
  return NArray::GenerateOne(size, randn_op);
}

NArray NArray::RandBernoulli(const Scale& size, float p) {
  CHECK_LE(p, 1);
  CHECK_LE(0, p);
  RandBernoulliOp* op = new RandBernoulliOp();
  op->closure = {p};
  return NArray::GenerateOne(size, op);
}

NArray NArray::LoadFromFile(const Scale& size, const string& fname, shared_ptr<IFileLoader> loader) {
  FileLoaderOp* loader_op = new FileLoaderOp();
  loader_op->closure = {fname, size, loader};
  return NArray::GenerateOne(size, loader_op);
}

NArray NArray::Zeros(const Scale& size) {
  return NArray::Constant(size, 0.0);
}

NArray NArray::Ones(const Scale& size) {
  return NArray::Constant(size, 1.0);
}

NArray NArray::MakeNArray(const Scale& size, shared_ptr<float> array) {
  ArrayLoaderOp* loader_op = new ArrayLoaderOp();
  loader_op->closure = {array};
  return NArray::GenerateOne(size, loader_op);
}

NArray PushGradAndPullWeight(const NArray & grad, const std::string & layer_name) {
  SyncWithPSOp * op = new SyncWithPSOp();
  op->closure = { layer_name };
  return NArray::ComputeOne({ grad }, grad.Size(), op);
}

// DAG building operations
vector<NArray> NArray::Compute(
    const vector<NArray>& params,
    const vector<Scale>& result_sizes,
    PhysicalComputeFn* fn) {
  auto& physical_dag = MinervaSystem::Instance().physical_dag();
  auto current_device_id = MinervaSystem::Instance().current_device_id_;
  auto rst = Map<NArray>(result_sizes, [&](const Scale& size) {
    return NArray(physical_dag.NewDataNode(PhysicalData(size, current_device_id, MinervaSystem::Instance().GenerateDataId())));
  });
  auto rst_data_nodes = Map<PhysicalDataNode*>(rst, [](const NArray& i) {
    return CHECK_NOTNULL(i.data_node_);
  });
  auto param_data_nodes = Map<PhysicalDataNode*>(params, [](const NArray& i) {
    return CHECK_NOTNULL(i.data_node_);
  });
  fn->device_id = current_device_id;
  MinervaSystem::Instance().physical_dag().NewOpNode(param_data_nodes, rst_data_nodes, {fn});
  return rst;
}

NArray NArray::ComputeOne(const vector<NArray>& params, const Scale& size, PhysicalComputeFn* fn) {
  auto& physical_dag = MinervaSystem::Instance().physical_dag();
  auto current_device_id = MinervaSystem::Instance().current_device_id_;
  auto rst = NArray(physical_dag.NewDataNode(PhysicalData(size, current_device_id, MinervaSystem::Instance().GenerateDataId())));
  auto param_data_nodes = Map<PhysicalDataNode*>(params, [](const NArray& i) {
    return CHECK_NOTNULL(i.data_node_);
  });
  fn->device_id = current_device_id;
  MinervaSystem::Instance().physical_dag().NewOpNode(param_data_nodes, {CHECK_NOTNULL(rst.data_node_)}, {fn});
  return rst;
}

NArray NArray::GenerateOne(const Scale& size, PhysicalComputeFn* fn) {
  return NArray::ComputeOne({}, size, fn);
}

// Constructors and destructors
NArray::NArray() : data_node_(nullptr) {}

NArray::NArray(const NArray& other) : data_node_(other.data_node_) {
  if (data_node_ != nullptr) {
    MinervaSystem::Instance().IncrExternRC(data_node_);
  }
}

NArray::NArray(NArray&& other) : data_node_(other.data_node_) {
  other.data_node_ = nullptr;
}

NArray& NArray::operator=(const NArray& other) {
  if (this == &other) {
    return *this;
  }
  if (data_node_ != nullptr) {
    MinervaSystem::Instance().DecrExternRC(data_node_);
  }
  data_node_ = other.data_node_;
  if (data_node_ != nullptr) {
    MinervaSystem::Instance().IncrExternRC(data_node_);
  }
  return *this;
}

NArray& NArray::operator=(NArray&& other) {
  if (this == &other) {
    return *this;
  }
  if (data_node_ != nullptr) {
    MinervaSystem::Instance().DecrExternRC(data_node_);
  }
  data_node_ = other.data_node_;
  other.data_node_ = nullptr;
  return *this;
}

NArray::~NArray() {
  if (data_node_ != nullptr && MinervaSystem::IsAlive()) {
    MinervaSystem::Instance().DecrExternRC(data_node_);
  }
}

// Matmult
NArray operator*(const NArray& lhs, const NArray& rhs) {
  CHECK_EQ(lhs.Size().NumDims(), 2) << "eligible only for 2D";
  CHECK_EQ(rhs.Size().NumDims(), 2) << "eligible only for 2D";
  CHECK_EQ(lhs.Size(1), rhs.Size(0)) << "size must match";
  Scale newsize = {lhs.Size(0), rhs.Size(1)};
  MatMultOp* matmult_op = new MatMultOp();
  return NArray::ComputeOne({lhs, rhs}, newsize, matmult_op);
}

NArray& NArray::operator*=(const NArray& rhs) {
  return *this = (*this * rhs);
}

// Shape
NArray NArray::Reshape(const Scale& dims) const {
  CHECK_EQ(Size().Prod(), dims.Prod()) << "dimension mismatch";
  return NArray::ComputeOne({*this}, dims, new ReshapeOp());
}

NArray NArray::Trans() const {
  CHECK_EQ(Size().NumDims(), 2) << "eligible only for 2D";
  Scale newsize = {Size(1), Size(0)};
  TransOp* trans_op = new TransOp();
  return NArray::ComputeOne({*this}, newsize, trans_op);
}

// Replicate matrix
NArray NArray::NormArithmetic(const NArray& rhs, ArithmeticType type) const {
  auto& lhs = *this;
  CHECK_EQ(lhs.Size().NumDims(), rhs.Size().NumDims()) << "#dimension mismatch";
  vector<int> dims_to_replicate;
  // Dimensions to replicate
  for (size_t i = 0; i < lhs.Size().NumDims(); ++i) {
    if (lhs.Size()[i] == rhs.Size()[i]) {
      continue;
    } else if (rhs.Size()[i] != 1) {
      LOG(FATAL) << "NormArithmetic cannot replicate a dimension that is not 1";
    } else {
      dims_to_replicate.push_back(i);
    }
  }
  CHECK_GT(dims_to_replicate.size(), 0) << "nothing to replicate";
  NormArithmeticOp* op = new NormArithmeticOp();
  op->closure.type = type;
  op->closure.dims_to_replicate = dims_to_replicate;
  return NArray::ComputeOne({lhs, rhs}, lhs.Size(), op);
}

// System
void NArray::WaitForEval() const {
  MinervaSystem::Instance().WaitForEval({*this});
}

void NArray::StartEval() const {
  MinervaSystem::Instance().StartEval({*this});
}

shared_ptr<float> NArray::Get() const {
  WaitForEval();
  return MinervaSystem::Instance().GetValue(*this);
}

void NArray::ToStream(ostream& out, const FileFormat& format) const {
  shared_ptr<float> ptr = Get();
  float* value = ptr.get();
  if (format.binary) {
    out.write(reinterpret_cast<char*>(value), Size().Prod() * sizeof(float));
  } else {
    for (int i = 0; i < Size().Prod(); ++i) {
      out << setprecision(4) << value[i] << "\t";
    }
  }
}

void NArray::ToFile(const std::string& filename, const FileFormat& format) const {
  ofstream fout(filename.c_str());
  ToStream(fout, format);
  fout.close();
}

NArray::NArray(PhysicalDataNode* node) : data_node_(node) {
  if (data_node_ != nullptr) {
    MinervaSystem::Instance().IncrExternRC(data_node_);
  }
}

}  // namespace minerva

