#include <fstream>
#include <memory>

#include "mb_loader.h"
#include "narray/narray.h"
#include "op/physical.h"
#include "op/logical.h"
#include "op/impl/basic.h"

using namespace std;

namespace minerva {

struct OneFileMBLoadClosure {
  std::string data_file_name;
  Scale load_shape;
  int sample_start_index;
};
class OneFileMBLoadOp :
  public LogicalDataGenFn,
  public PhysicalComputeFn,
  public ClosureTrait<OneFileMBLoadClosure> {
 public:
  void Execute(DataList& inputs, DataList& outputs, ImplType impl_type) {
    CHECK_EQ(impl_type, ImplType::kBasic) << "mb loader operator only has basic implementation";
    ifstream fin(closure.data_file_name.c_str(), ios::binary);
    int num_samples, sample_length;
    fin.read(reinterpret_cast<char*>(&num_samples), sizeof(int));
    fin.read(reinterpret_cast<char*>(&sample_length), sizeof(int));
    int num_floats = closure.load_shape.Prod();
    float* buf = new float[num_floats];
    // read contents
    int read_start = closure.sample_start_index * sample_length;
    fin.seekg(read_start * sizeof(float), ios::cur);
    int num_floats_read = 0;
    const int num_floats_file = num_samples * sample_length;
    while(num_floats_read < num_floats) {
      int to_read = min(num_floats - num_floats_read, num_floats_file - read_start);
      fin.read(reinterpret_cast<char*>(buf + num_floats_read), to_read * sizeof(float));
      num_floats_read += to_read;
      fin.seekg(2 * sizeof(int), ios::beg);
      read_start = 0;
    }
    // split
    size_t numdims = outputs[0].Size().NumDims();
    Scale dststart = Scale::Origin(numdims);
    for(DataShard& ds : outputs) {
      basic::NCopy(buf, closure.load_shape, ds.Offset(), ds.GetCpuData(), ds.Size(), dststart, ds.Size());
    }
    delete [] buf;
    fin.close();
  }
  NVector<Chunk> Expand(const NVector<Scale>& partsizes) {
    OneFileMBLoadOp* mbload_op = new OneFileMBLoadOp(*this);
    const vector<Chunk>& retchs = Chunk::Compute({}, partsizes.ToVector(), mbload_op);
    return NVector<Chunk>(retchs, partsizes.Size());
  }
  std::string Name() const {
    std::stringstream ss;
    ss << "load-mb@" << closure.sample_start_index;
    return ss.str();
  }
};
OneFileMBLoader::OneFileMBLoader(const string& name, const Scale& s):
  data_file_name_(name), sample_shape_(s), sample_start_index_(0) {
  partition_shapes_per_sample_ = sample_shape_.ToNVector();
  ifstream fin(name.c_str(), ios::binary);
  fin.read(reinterpret_cast<char*>(&num_samples_), 4);
  fin.read(reinterpret_cast<char*>(&sample_length_), 4);
  fin.close();
  CHECK_EQ(sample_shape_.Prod(), sample_length_) << "size of each sample mismatch!";
}
OneFileMBLoader::~OneFileMBLoader() {
}
NArray OneFileMBLoader::LoadNext(int stepsize) {
  Scale load_shape = sample_shape_.Concat(stepsize);
  OneFileMBLoadOp* op = new OneFileMBLoadOp;
  op->closure = {data_file_name_, load_shape, sample_start_index_};
  NVector<Scale> partition_shapes = partition_shapes_per_sample_.Map<Scale>(
      [stepsize] (const Scale& s) { return s.Concat(stepsize); }
      );
  partition_shapes.AugmentDim();
  sample_start_index_ = (sample_start_index_ + stepsize) % num_samples_;
  return NArray::Generate(load_shape, op, partition_shapes);
}

} // end of namespace minerva
