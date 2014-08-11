#pragma once
#include <string>
#include <vector>
#include "../narray.h"

namespace minerva {

class NArray;

class OneFileMBLoader {
 public:
  OneFileMBLoader(const std::string&, const Scale& sample_shape);
  ~OneFileMBLoader();
  virtual NArray LoadNext(int stepsize);
  void set_partition_shapes_per_sample(const NVector<Scale>& ps) { partition_shapes_per_sample_ = ps; }
  int num_samples() const { return num_samples_; }

 protected:
  // these members should not be modified once initialized
  const std::string data_file_name_;
  const Scale sample_shape_;
  int num_samples_, sample_length_;

  // these are mutable members
  NVector<Scale> partition_shapes_per_sample_;
  int sample_start_index_;
};

}
