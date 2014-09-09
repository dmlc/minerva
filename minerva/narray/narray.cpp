#include "narray.h"
#include "common/common.h"

using namespace std;

namespace minerva {

static MinervaSystem& ms = MinervaSystem::Instance();

NArray NArray::Constant(const Scale& size, float val) {
  FillOp* fill_op = new FillOp;
  fill_op->closure = {val};
  return NArray::Generate(size, fill_op, parts);
}

////////////////////////////////////////////////////
// constructors & destructors
////////////////////////////////////////////////////
// public
NArray::NArray(): data_node_(nullptr) {}
NArray::NArray(const NArray& other): data_node_(other.data_node_) {
  if(data_node_ != nullptr)
    ms.IncrExternRC(data_node_);
}
NArray::~NArray() {
  if(data_node_ != nullptr)
    ms.IncrExternRC(data_node_, -1);
}
NArray& NArray::operator = (const NArray& other) {
  auto old_dnode = data_node_;
  data_node_ = other.data_node_;
  if(data_node_ != nullptr)
    ms.IncrExternRC(data_node_, 1);
  if(old_dnode != nullptr)
    ms.IncrExternRC(old_dnode, -1);
  return *this;
}
// private
NArray::NArray(LogicalDataNode* node): data_node_(node) {
  ms.IncrExternRC(data_node_);
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
    auto rst_node = physical_dag.NewDataNode(PhysicalData(size, device_info, data_store.GenerateDataID()));
  std::vector<NArray> rst;
  std::vector<LogicalDataNode*> rst_data_nodes;
  for(Scale size : result_sizes) {
    LogicalData ldata(size);
    ldata.device_info = device_info;
    LogicalDataNode* rst_node = ldag.NewDataNode(ldata);
    rst.push_back(NArray(rst_node));
    rst_data_nodes.push_back(rst_node);
  }
  std::vector<LogicalDataNode*> param_data_nodes;
  for(NArray p : params) {
    param_data_nodes.push_back(p.data_node_);
  }
  fn->device_info = device_info;
  ldag.NewOpNode(param_data_nodes, rst_data_nodes, {fn});
  return rst;
}

NArray NArray::Generate(const Scale& size, LogicalDataGenFn* fn) {
  LogicalDag& ldag = MinervaSystem::Instance().logical_dag();
  DeviceInfo device_info = MinervaSystem::Instance().device_info();
  fn->device_info = device_info;
  LogicalData ldata(size, fn);
  ldata.partitions = parts;
  ldata.device_info = device_info;
  LogicalDataNode* rst_node = ldag.NewDataNode(ldata);
  return NArray(rst_node);
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

NArray NArray::LoadFromFile(const Scale& size, const std::string& fname,
    IFileLoader* loader, const Scale& numparts) {
  FileLoaderOp* loader_op = new FileLoaderOp;
  loader_op->closure = {fname, size, loader};
  return NArray::Generate(size, loader_op, numparts);
}

NArray NArray::MakeNArray(const Scale& size, std::shared_ptr<float> array, const Scale& numparts) {
  ArrayLoaderOp* loader_op = new ArrayLoaderOp;
  loader_op->closure = {array, size};
  return NArray::Generate(size, loader_op, numparts);
}

}

