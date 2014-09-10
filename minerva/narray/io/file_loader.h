#pragma once
#include "op/physical.h"
#include "op/logical.h"
#include "op/context.h"

namespace minerva {

class IFileLoader {
 public:
  virtual void Load(const std::string& fname, const Scale& size, DataList& out_shards) {}  // Not pure virtual for Python
  virtual ~IFileLoader() {}
};

struct FileLoaderClosure {
  std::string fname;
  Scale size;
  IFileLoader* loader;
};

class FileLoaderOp :
  public LogicalDataGenFn,
  public PhysicalComputeFn,
  public ClosureTrait<FileLoaderClosure> {
 public:
  void Execute(DataList& inputs, DataList& outputs, const Context&);
  NVector<Chunk> Expand(const NVector<Scale>& part_sizes);
  std::string Name() const;
};

class SimpleFileLoader : public IFileLoader {
 public:
  virtual void Load(const std::string& fname, const Scale& size, DataList& out_shards);
};

}
