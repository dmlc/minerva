#pragma once
#include <vector>
#include <memory>
#include "common/scale.h"

namespace minerva {

class ComputeFn;
class MData {
 public:
  virtual ~MData() = default;
  virtual const Scale& shape() const = 0;
};

class IBackend {
 public:
  virtual ~IBackend() = default;
  virtual std::vector<MData*> Create(const std::vector<MData*>& params, const std::vector<Scale>& result_sizes, ComputeFn* fn) = 0;
  MData* CreateOne(MData* param, const Scale& result_size, ComputeFn* fn) { return Create({param}, {result_size}, fn)[0]; }
  //virtual MData* RecordCreateInplace(MData* param, ComputeFn* fn) = 0;
  virtual void ShallowCopy(MData*& to, MData* from) = 0;
  virtual void Destroy(MData* ) = 0;
  virtual void Issue(MData* ) = 0;
  virtual void Wait(MData* ) = 0;
  //virtual void Wait(const std::vector<MData*>& ) = 0;
  virtual void WaitForAll() = 0;
  virtual std::shared_ptr<float> GetValue(MData* ) = 0;
};

}
