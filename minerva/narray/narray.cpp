#include "narray.h"
#include "op/logical_op.h"
#include "system/minerva_system.h"
#include "io/file_loader.h"
#include <fstream>
#include <iomanip>

using namespace std;

namespace minerva {

NArray::NArray(): data_node_(NULL) {}
NArray::NArray(LogicalDataNode* node): data_node_(node) {}
NArray::~NArray() {}

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
  
NArray NArray::Generate(const Scale& size, LogicalDataGenFn* fn, const NVector<PartInfo>& parts) {
  LogicalDag& ldag = MinervaSystem::Instance().logical_dag();
  LogicalDataNode* rst_node = ldag.NewDataNode({size, fn, parts});
  return NArray(rst_node);
}

NArray NArray::Generate(const Scale& size, LogicalDataGenFn* fn, const Scale& numparts) {
  NVector<Scale> partsizes = size.EquallySplit(numparts);
  return Generate(size, fn, 
      partsizes.Map<PartInfo>(
        [] (const Scale& size) { return PartInfo{kUnknownPlace, size}; }
      )
   );
}

NArray NArray::Constant(const Scale& size, float val, const NVector<PartInfo>& parts) {
  FillOp* fill_op = new FillOp;
  fill_op->closure = {val};
  return NArray::Generate(size, fill_op, parts);
}

NArray NArray::Randn(const Scale& size, float mu, float var, const NVector<PartInfo>& parts) {
  RandnOp* randn_op = new RandnOp;
  randn_op->closure = {mu, var};
  return NArray::Generate(size, randn_op, parts);
}

NArray NArray::Constant(const Scale& size, float val, const Scale& numparts) {
  FillOp* fill_op = new FillOp;
  fill_op->closure = {val};
  return NArray::Generate(size, fill_op, numparts);
}

NArray NArray::Randn(const Scale& size, float mu, float var, const Scale& numparts) {
  RandnOp* randn_op = new RandnOp;
  randn_op->closure = {mu, var};
  return NArray::Generate(size, randn_op, numparts);
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


NArray NArray::RePartition(const NVector<PartInfo>& partitions) {
  if(partitions == data_node_->data_.partitions) {
    // partition is the same
    return *this;
  }
  // new partition plan
  Scale total_size = Scale::Merge( 
      partitions.Map<Scale>( [] (const PartInfo& pi) { return pi.size; } )
    );
  assert(total_size == data_node_->data_.size); // validity
  PartitionOp* part_op = new PartitionOp;
  part_op->closure = {partitions};
  return Compute({*this}, {this->Size()}, part_op)[0];
}

void NArray::Eval() {
  MinervaSystem::Instance().Eval(*this);
}

float* NArray::Get() {
  Eval();
  return MinervaSystem::Instance().GetValue(*this);
}

void NArray::ToFile(const std::string& filename, const FileFormat& format) {
  float* value = Get();
  ofstream fout(filename.c_str());
  if(format.binary) {
    fout.write(reinterpret_cast<char*>(value), Size().Prod() * sizeof(float));
  }
  else {
    for(int i = 0; i < Size().Prod(); ++i)
      fout << setprecision(4) << value[i] << "\t";
  }
  fout.close();
}

NArray NArray::LoadFromFile(const Scale& size, const std::string& fname,
    IFileLoader* loader, const Scale& numparts) {
  FileLoaderOp* loader_op = new FileLoaderOp;
  loader_op->closure = {fname, size, loader};
  return NArray::Generate(size, loader_op, numparts);
}

} // end of namespace minerva
