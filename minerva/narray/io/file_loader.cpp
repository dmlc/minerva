#include <fstream>
#include <glog/logging.h>

#include "file_loader.h"
#include "op/impl/basic.h"

using namespace std;

namespace minerva {

void FileLoaderOp::Execute(DataList& inputs, DataList& outputs, ImplType impl_type) {
  CHECK_EQ(impl_type, ImplType::kBasic) << "file loader operator only has basic implementation";
  closure.loader->Load(closure.fname, closure.size, outputs);
}

NVector<Chunk> FileLoaderOp::Expand(const NVector<Scale>& part_sizes) {
  FileLoaderOp* op = new FileLoaderOp(*this);
  const vector<Chunk>& ret_chunks = Chunk::Compute({}, part_sizes.ToVector(), op);
  return NVector<Chunk>(ret_chunks, part_sizes.Size());
}

string FileLoaderOp::Name() const {
  stringstream ss;
  ss << "load(" << closure.fname << ")";
  return ss.str();
}

void SimpleFileLoader::Load(const string& fname, const Scale& size, DataList& out_shards) {
  size_t numvalue = size.Prod();
  float* ptr = new float[numvalue]; // TODO should use data_store
  ifstream fin(fname.c_str());
  fin.read(reinterpret_cast<char*>(ptr), numvalue * sizeof(float));
  fin.close();
  // partition the file content
  size_t numdims = size.NumDims();
  Scale dststart = Scale::Origin(numdims);
  for(DataShard& ds : out_shards) {
    basic::NCopy(ptr, size, ds.Offset(), ds.GetCpuData(), ds.Size(), dststart, ds.Size());
  }
  delete[] ptr;
}

} // end of namespace minerva
