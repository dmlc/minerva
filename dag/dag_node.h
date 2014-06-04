#pragma once
#include <cstdint>
#include <vector>
#include <set>
#include <initializer_list>
#include <functional>
#include <mutex>

#include "common/common.h"
#include "common/index.h"
#include "dag_context.h"

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
  DataNodeMeta(const Index& size, const Index& off,
      const Index& chidx):
    size(size), offset(off), chunk_index(chidx) {
    length = size.Prod();
  }
  DataNodeMeta(const Index& size): size(size) {
    length = size.Prod();
    offset = Index::Origin(size.NumDims());
    chunk_index = Index::Origin(size.NumDims());
  }
  size_t length;
  Index size, offset, chunk_index;
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
  typedef std::function<void()> Runner;
  OpNode();
  ~OpNode();
  void set_runner(const Runner& r) { runner_ = r; }
  const Runner& runner() const { return runner_; };
  void set_context(const OpNodeContext& ctx) { context_ = ctx; }
  const OpNodeContext& context() const { return context_; }

  NodeTypes Type() const { return OP_NODE; }

 private:
  Runner runner_;
  //std::vector<DataNode*> inputs_, outputs_; TODO not sure we need this or not
  OpNodeContext context_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

} // end of namespace minerva
