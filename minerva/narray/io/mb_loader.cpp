#include <fstream>
#include <memory>

#include "mb_loader.h"
#include "narray/narray.h"
#include "op/physical.h"
#include "op/logical.h"

using namespace std;

namespace minerva {

struct OneFileMBLoadClosure {
  std::shared_ptr<ifstream> fin_ptr;
  int sample_start_index;
  int step_size;
};
class OneFileMBLoadOp :
  public LogicalDataGenFn,
  public PhysicalComputeFn,
  public ClosureTrait<OneFileMBLoadClosure> {
 public:
  void Execute(DataList& inputs, DataList& outputs, ImplType impl_type) {
  }
  NVector<Chunk> Expand(const NVector<Scale>& part_sizes) {
  }
  std::string Name() const {
    std::stringstream ss;
    ss << "load-mb@" << closure.sample_start_index;
    return ss.str();
  }
};
OneFileMBLoader::OneFileMBLoader(const string& name):
  data_file_name_(name), fin_ptr_(new ifstream(name.c_str()), ios::binary),
  sample_start_index_(0) {
  fin_ptr_->read(reinterpret_cast<char*>(&num_samples_), 4);
  fin_ptr_->read(reinterpret_cast<char*>(&sample_length_), 4);
}
OneFileMBLoader::~OneFileMBLoader() {
}

void OneFileMBLoader::LoadNext(int stepsize) {
  sample_start_index_ = (sample_start_index_ + stepsize) % num_samples_;
}
NArray OneFileMBLoader::GetData() {
  // TODO
  return NArray();
}
NArray OneFileMBLoader::GetLabel() {
  // TODO
  return NArray();
}

} // end of namespace minerva
