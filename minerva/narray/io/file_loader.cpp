#include <fstream>
#include <glog/logging.h>

#include "file_loader.h"

using namespace std;

namespace minerva {

void FileLoaderOp::Execute(DataList& inputs, DataList& outputs, IMPL_TYPE impl_type) {
  CHECK_EQ(impl_type, BASIC) << "file loader operator only has basic implementation";
  closure.loader->Load(closure.fname, outputs);
}

NVector<Chunk> FileLoaderOp::Expand(const NVector<Scale>& part_sizes) {
  FileLoaderOp* op = new FileLoaderOp;
  op->closure = closure;
  const std::vector<Chunk>& ret_chunks = Chunk::Compute({}, part_sizes.ToVector(), op);
  return NVector<Chunk>(ret_chunks, part_sizes.Size());
}

std::string FileLoaderOp::Name() const {
  std::stringstream ss;
  ss << "load(\"" << closure.fname << "\")";
  return ss.str();
}

void SimpleFileLoader::Load(const std::string& fname, DataList& out_shards) {
  int totalsize = 0;
  for(DataShard& ds : out_shards) {
    totalsize += ds.Size().Prod();
  }
  float* total_contents = new float[totalsize]; // TODO should allocate use data_store
  ifstream fin(fname.c_str());
  fin.read(reinterpret_cast<char*>(total_contents), totalsize * sizeof(float));
  fin.close();
  // partition the file content
}

} // end of namespace minerva
