#pragma once
#include <cstdint>
#include <vector>
#include <set>
#include <initializer_list>
#include <functional>
#include <mutex>

#include "common/common.h"
#include "common/scale.h"
#include "dag/dag_context.h"
#include "dag/op/closure.h"

namespace minerva {

class DagNode;
struct DataNodeMeta;
class DataNode;
class OpNode;

class DagNode {
  friend class Dag;
  friend class DagEngine;

 public:
  enum NodeTypes {
    OP_NODE = 0,
    DATA_NODE
  };
  DagNode();
  virtual ~DagNode();
  void AddParent(DagNode*);
  void AddParents(std::initializer_list<DagNode*>);
  bool DeleteParent(DagNode*);
  // getters
  uint64_t node_id() { return node_id_; };

  virtual NodeTypes Type() const = 0;

 protected:
  uint64_t node_id_;
  std::set<DagNode*> successors_;
  std::set<DagNode*> predecessors_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DagNode);
};

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
  DataNode() { Init(); }
  DataNode(const DataNodeMeta& meta): meta_(meta) { Init(); }
  ~DataNode() {}

  void set_data_id(uint64_t id) { data_id_ = id; }
  uint64_t data_id() const { return data_id_; }
  void set_meta(const DataNodeMeta& meta) { meta_ = meta; }
  const DataNodeMeta& meta() const { return meta_; }
  void set_context(const DataNodeContext& ctx) { context_ = ctx; }
  const DataNodeContext& context() const { return context_; }

  NodeTypes Type() const { return DATA_NODE; }

 private:
  uint64_t data_id_;
  DataNodeMeta meta_;
  DataNodeContext context_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataNode);
  void Init();
};

class OpNode: public DagNode {
 public:
  OpNode();
  ~OpNode();
  void set_closure(void* r) { closure_ = r; }
  void* closure() { return closure_; };
  void set_context(const OpNodeContext& ctx) { context_ = ctx; }
  const OpNodeContext& context() const { return context_; }
  void set_inputs(const std::vector<DataNode*>& in) { inputs_ = in; }
  const std::vector<DataNode*>& inputs() { return inputs_; }
  void set_outputs(const std::vector<DataNode*>& out) { outputs_ = out; }
  const std::vector<DataNode*>& outputs() { return outputs_; }

  NodeTypes Type() const { return OP_NODE; }

 private:
  void* closure_;
  OpNodeContext context_;
  std::vector<DataNode*> inputs_, outputs_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

} // end of namespace minerva
