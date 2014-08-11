#pragma once
#include <string>
#include <vector>
#include "../narray.h"

namespace minerva {

class NArray;

class IMiniBatchLoader {
 public:
  virtual void LoadNext(int stepsize) = 0;
  virtual NArray GetData() = 0;
  virtual NArray GetLabel() = 0;
};

class OneFileMBLoader : public IMiniBatchLoader {
 public:
  OneFileMBLoader(const std::string& );
  ~OneFileMBLoader();
  virtual void LoadNext(int stepsize);
  virtual NArray GetData();
  virtual NArray GetLabel();
  int num_samples() const { return num_samples_; }
 protected:
  std::string data_file_name_;
  std::shared_ptr<std::ifstream> fin_ptr_;
  int sample_start_index_;
  int num_samples_, sample_length_;
};

}
