#include "narray.h"
#include "common/common.h"
#include "system/minerva_system.h"
#include "io/file_loader.h"
#include "io/array_loader.h"
#include <fstream>
#include <glog/logging.h>
#include <iomanip>

using namespace std;

namespace minerva {

static MinervaSystem& ms = MinervaSystem::Instance();

// Static constructors
NArray NArray::Constant(const Scale& size, float val) {
  FillOp* fill_op = new FillOp;
  fill_op->closure = {val};
  return NArray::GenerateOne(size, fill_op);
}

NArray NArray::Randn(const Scale& size, float mu, float var) {
  RandnOp* randn_op = new RandnOp();
  randn_op->closure = {mu, var};
  return NArray::GenerateOne(size, randn_op);
}

NArray NArray::LoadFromFile(const Scale& size, const std::string& fname, IFileLoader* loader) {
  FileLoaderOp* loader_op = new FileLoaderOp;
  loader_op->closure = {fname, size, loader};
  return NArray::GenerateOne(size, loader_op);
}
<<<<<<< HEAD

NArray NArray::Zeros(const Scale& size) {
  return NArray::Constant(size, 0.0);
}

NArray NArray::Ones(const Scale& size) {
  return NArray::Constant(size, 1.0);
}

NArray NArray::MakeNArray(const Scale& size, shared_ptr<float> array) {
  ArrayLoaderOp* loader_op = new ArrayLoaderOp;
  loader_op->closure = {array, size};
  return NArray::GenerateOne(size, loader_op);
}

// DAG building operations
vector<NArray> NArray::Compute(
    const vector<NArray>& params,
    const vector<Scale>& result_sizes,
    PhysicalComputeFn* fn) {
  auto& physical_dag = MinervaSystem::Instance().physical_dag();
  auto& data_store = MinervaSystem::Instance().data_store();
  auto device_info = MinervaSystem::Instance().device_info();
  auto rst = Map<NArray>(result_sizes, [&](const Scale& scale) {
    return NArray(physical_dag.NewDataNode(PhysicalData(size, device_info, data_store.GenerateDataId())));
  });
  auto rst_data_nodes = Map<PhysicalDataNode*>(rst, [](const NArray& i) {
    return i.data_node();
  });
  auto param_data_nodes = Map<PhysicalDataNode*>(params, [](const NArray& i) {
    return i.data_node();
  });
  fn->device_info = device_info;
=======
// private
NArray::NArray(LogicalDataNode* node): data_node_(node) {
  ms.IncrExternRC(data_node_);
}

////////////////////////////////////////////////////
// computation methods
////////////////////////////////////////////////////
std::vector<NArray> NArray::Compute(std::vector<NArray> params,
    std::vector<Scale> result_sizes, LogicalComputeFn* fn) {
  LogicalDag& ldag = MinervaSystem::Instance().logical_dag();
  uint64_t device_id = MinervaSystem::Instance().device_id();
  std::vector<NArray> rst;
  std::vector<LogicalDataNode*> rst_data_nodes;
  for(Scale size : result_sizes) {
    LogicalData ldata(size);
    ldata.device_id = device_id;
    LogicalDataNode* rst_node = ldag.NewDataNode(ldata);
    rst.push_back(NArray(rst_node));
    rst_data_nodes.push_back(rst_node);
  }
  std::vector<LogicalDataNode*> param_data_nodes;
  for(NArray p : params) {
    param_data_nodes.push_back(p.data_node_);
  }
  fn->device_id = device_id;
>>>>>>> master
  ldag.NewOpNode(param_data_nodes, rst_data_nodes, {fn});
  return rst;
}

<<<<<<< HEAD
NArray NArray::ComputeOne(const vector<NArray>& params, const Scale& size, PhysicalComputeFn* fn) {
  return NArray::Compute(params, {size}, fn)[0];
=======
NArray NArray::Generate(const Scale& size, LogicalDataGenFn* fn, const NVector<Scale>& parts) {
  LogicalDag& ldag = MinervaSystem::Instance().logical_dag();
  int64_t device_id = MinervaSystem::Instance().device_id();
  fn->device_id = device_id;
  LogicalData ldata(size, fn);
  ldata.partitions = parts;
  ldata.device_id = device_id;
  LogicalDataNode* rst_node = ldag.NewDataNode(ldata);
  return NArray(rst_node);
>>>>>>> master
}

NArray NArray::GenerateOne(const Scale& size, PhysicalComputeFn* fn) {
  return NArray::ComputeOne({}, size, fn);
}

// Constructors and destructors
NArray::NArray() : data_node_(nullptr) {}

NArray::NArray(const NArray& other) : data_node_(other.data_node_) {
  if (data_node_ != nullptr) {
    ms.IncrExternRC(data_node_);
  }
}

NArray::NArray(NArray&& other) : data_node_(other.data_node_) {
  other.data_node_ = 0;
}

NArray& NArray::operator=(const NArray& other) {
  if (this == &other) {
    return *this;
  }
  if (data_node_ != nullptr) {
    ms.IncrExternRC(data_node_, -1);
  }
  data_node_ = other.data_node_;
  if (data_node_ != nullptr) {
    ms.IncrExternRC(data_node_);
  }
  return *this;
}

NArray& NArray::operator=(NArray&& other) {
  if (this == &other) {
    return *this;
  }
  if (data_node_ != nullptr) {
    ms.IncrExternRC(data_node_, -1);
  }
  data_node_ = other.data_node_;
  other.data_node_ = nullptr;
  return *this;
}

NArray::~NArray() {
  if (data_node_ != nullptr) {
    ms.IncrExternRC(data_node_, -1);
  }
}

// Matmult
NArray operator*(NArray lhs, NArray rhs) {
  CHECK_EQ(lhs.Size().NumDims(), 2) << "eligible only for 2D";
  CHECK_EQ(rhs.Size().NumDims(), 2) << "eligible only for 2D";
  CHECK_EQ(lhs.Size(1), rhs.Size(0)) << "size must match";
  Scale newsize = {lhs.Size(0), rhs.Size(1)};
  MatMultOp* matmult_op = new MatMultOp;
  return NArray::ComputeOne({lhs, rhs}, newsize, matmult_op);
}

// Shape
NArray NArray::Reshape(const Scale& dims) const {
  // TODO
  CHECK(false) << "not implemented";
  return NArray();
}

NArray NArray::Trans() const {
  CHECK_EQ(Size().NumDims(), 2) << "eligible only for 2D";
  Scale newsize = {Size(1), Size(0)};
  TransOp* trans_op = new TransOp;
  return NArray::ComputeOne({*this}, newsize, trans_op);
}

// Replicate matrix
NArray NArray::NormArithmetic(NArray rhs, ArithmeticType type) const {
  auto& lhs = *this;
  CHECK_EQ(lhs.Size().NumDims(), rhs.Size().NumDims()) << "NormArithmetic #dimension mismatch";
  vector<int> dims_to_replicate;
  // Dimensions to replicate
  for (size_t i = 0; i < lhs.Size().NumDims(); ++i) {
    if (lhs.Size()[i] == rhs.Size()[i]) {
      continue;
    } else if (rhs.Size()[i] != 1) {
      CHECK(false) << "NormArithmetic cannot replicate a dimension that is not 1";
    } else {
      dims_to_replicate.push_back(i);
    }
  }
  NormArithmeticOp* op = new NormArithmeticOp;
  op->closure.type = type;
  op->closure.dims_to_replicate = dims_to_replicate;
  return NArray::ComputeOne({lhs, rhs}, lhs.Size(), op);
}

// System
void NArray::Eval() const {
  MinervaSystem::Instance().Eval({*this});
}

void NArray::EvalAsync() const {
  MinervaSystem::Instance().EvalAsync({*this});
}

float* NArray::Get() const {
  Eval();
  return MinervaSystem::Instance().GetValue(*this);
}

void NArray::ToStream(ostream& out, const FileFormat& format) const {
  float* value = Get();
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
  if (data_node != nullptr) {
    ms.IncrExternRC(data_node_);
  }
}

}

