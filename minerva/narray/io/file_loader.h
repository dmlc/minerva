#pragma once
#include <memory>
#include "op/physical.h"
#include "op/physical_fn.h"
#include "op/context.h"

namespace minerva {

class IFileLoader {
 public:
  virtual void Load(const std::string& fname, const Scale& size, const DataList& out_shards) {}  // Not pure virtual for Python
  virtual ~IFileLoader() {}
};

struct FileLoaderClosure {
  std::string fname;
  Scale size;
  std::shared_ptr<IFileLoader> loader;
};

class FileLoaderOp :
  public PhysicalComputeFn,
  public ClosureTrait<FileLoaderClosure> {
 public:
  void Execute(const DataList&, const DataList&, const Context&);
  std::string Name() const;
};

class SimpleFileLoader : public IFileLoader {
 public:
  virtual void Load(const std::string& fname, const Scale& size, const DataList& out_shards);
};

}

