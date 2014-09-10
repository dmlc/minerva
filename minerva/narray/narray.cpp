#include "narray.h"
#include "common/common.h"

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
  ldag.NewOpNode(param_data_nodes, rst_data_nodes, {fn});
  return rst;
}

NArray NArray::ComputeOne(const vector<NArray>& params, const Scale& size, PhysicalComputeFn* fn) {
  return NArray::Compute(params, {size}, fn)[0];
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

NArray& NArray::operator=(const NArray& other) {
  if (this == &other) {
    return *this;
  }
  auto old_dnode = data_node_;
  data_node_ = other.data_node_;
  if (data_node_ != nullptr) {
    ms.IncrExternRC(data_node_);
  }
  if (old_dnode != nullptr) {
    ms.IncrExternRC(old_dnode, -1);
  }
  return *this;
}

NArray::~NArray() {
  if (data_node_ != nullptr) {
    ms.IncrExternRC(data_node_, -1);
  }
}

// Operations
NArray operator*(NArray lhs, NArray rhs) {
  // validity
  assert(lhs.Size().NumDims() == 2 && rhs.Size().NumDims() == 2);
  assert(lhs.Size(1) == rhs.Size(0));
  Scale newsize = {lhs.Size(0), rhs.Size(1)};
  MatMultOp* matmult_op = new MatMultOp;
  return NArray::Compute({lhs, rhs}, {newsize}, matmult_op)[0];
}

// shape
NArray NArray::Reshape(const Scale& dims) {
  // TODO
  return NArray();
}

// Replicate matrix
NArray NArray::NormArithmetic(NArray rhs, ArithmeticType type) {
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
  return NArray::Compute({lhs, rhs}, {lhs.Size()}, op)[0];
}

NArray NArray::Trans() {
  // validity
  assert(Size().NumDims() == 2);
  Scale newsize = {Size(1), Size(0)};
  TransOp* trans_op = new TransOp;
  return NArray::Compute({*this}, {newsize}, trans_op)[0];
}


NArray NArray::RePartition(const NVector<Scale>& partitions) {
  /*if(partitions == data_node_->data_.partitions) {
    // partition is the same
    return *this;
  }
  // new partition plan
  Scale total_size = Scale::Merge(partitions);
  assert(total_size == data_node_->data_.size); // validity
  PartitionOp* part_op = new PartitionOp;
  part_op->closure = {partitions};
  return Compute({*this}, {this->Size()}, part_op)[0];*/
  //TODO
  return NArray();
}

void NArray::Eval() {
  MinervaSystem::Instance().Eval({*this});
}

void NArray::EvalAsync() {
  MinervaSystem::Instance().EvalAsync({*this});
}

float* NArray::Get() {
  Eval();
  return MinervaSystem::Instance().GetValue(*this);
}

void NArray::ToStream(ostream& out, const FileFormat& format) {
  float* value = Get();
  if(format.binary) {
    out.write(reinterpret_cast<char*>(value), Size().Prod() * sizeof(float));
  } else {
    for (int i = 0; i < Size().Prod(); ++i) {
      out << setprecision(4) << value[i] << "\t";
    }
  }
}

void NArray::ToFile(const std::string& filename, const FileFormat& format) {
  ofstream fout(filename.c_str());
  ToStream(fout, format);
  fout.close();
}

}

NArray::NArray(PhysicalDataNode* node): data_node_(node) {
  if (data_node != nullptr) {
    ms.IncrExternRC(data_node_);
  }
}

