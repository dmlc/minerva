#pragma once
#include <cstdint>
#include <vector>
#include <set>
#include <initializer_list>
#include <functional>
#include <mutex>

#include "common/common.h"
#include "common/scale.h"
#include "common/nvector.h"
#include "dag/dag_context.h"

namespace minerva {

class DagNode;
struct DataNodeMeta;
class DataNode;
class OpNode;


struct DataNodeMeta {
  DataNodeMeta(): length(0) {}
  DataNodeMeta(const DataNodeMeta& other):
    length(other.length), size(other.size),
    offset(other.offset), chunk_index(other.chunk_index) {}
  DataNodeMeta(const Scale& size, const Scale& off, const Scale& chidx):
    size(size), offset(off), chunk_index(chidx) {
      length = size.Prod();
  }
  DataNodeMeta(const Scale& size): size(size) {
    length = size.Prod();
    offset = Scale::Origin(size.NumDims());
    chunk_index = Scale::Origin(size.NumDims());
  }
  size_t length;
  Scale size, offset, chunk_index;
};

class DataNode: public DagNode {
 public:
  DataNode() {}
  DataNode(const DataNodeMeta& meta): meta_(meta) {}
  ~DataNode() {}

  //void set_data_id(uint64_t id) { data_id_ = id; }
  //uint64_t data_id() const { return data_id_; }
  void set_meta(const DataNodeMeta& meta) { meta_ = meta; }
  const DataNodeMeta& meta() const { return meta_; }
  void set_context(const DataNodeContext& ctx) { context_ = ctx; }
  const DataNodeContext& context() const { return context_; }

  NodeTypes Type() const { return DATA_NODE; }

 protected:
  //uint64_t data_id_; //TODO shall we have data_id ? The problem is logical 
                       //data_node has no data_id while physical node has
  DataNodeMeta meta_;
  DataNodeContext context_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataNode);
};

class LDagDataNode : public DataNode {
 public:
  LDagDataNode() {}
 private:
  NVector<DataNode*> mapped_pdag_nodes_;
};

class OpNodeRunner {
 public:
  virtual void Init() = 0;
  virtual void Compute(void* closure, std::vector<DataNode*> inputs,
      std::vector<DataNode*> outputs) = 0;
  virtual void Destroy() = 0;
};

class OpNode: public DagNode {
 public:
  OpNode();
  ~OpNode();
  void set_closure(void* r) { closure_ = r; }
  void* closure() { return closure_; };
  //void set_context(const OpNodeContext& ctx) { context_ = ctx; }
  //const OpNodeContext& context() const { return context_; }
  void set_inputs(const std::vector<DataNode*>& in) { inputs_ = in; }
  const std::vector<DataNode*>& inputs() { return inputs_; }
  void set_outputs(const std::vector<DataNode*>& out) { outputs_ = out; }
  const std::vector<DataNode*>& outputs() { return outputs_; }

  NodeTypes Type() const { return OP_NODE; }

 private:
  void* closure_;
  OpNodeRunner* runner_;
  //OpNodeContext context_;
  std::vector<DataNode*> inputs_, outputs_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

} // end of namespace minerva
