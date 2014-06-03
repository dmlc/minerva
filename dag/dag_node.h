#pragma once
#include <cstdint>
#include <vector>
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
  ~DagNode();
  void AddParent(DagNode*);
  void AddParents(std::initializer_list<DagNode*>);
  bool DeleteParent(DagNode*);
  // setters
  void set_context(const DagNodeContext& ctx) { context_ = ctx; }
  // getters
  uint64_t node_id() { return node_id_; };

  virtual NodeTypes Type() const = 0;

 protected:
  uint64_t node_id_;
  std::vector<DagNode*> successors_;
  std::vector<DagNode*> predecessors_;
  DagNodeContext context_;

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
  DataNodeMeta& meta() { return meta_; }
  const DataNodeMeta& meta() const { return meta_; }

  NodeTypes Type() const { return DATA_NODE; }
  bool CreateCPUData();
  bool CreateGPUData();
  float* GetCPUData();
  float* GetGPUData();

 private:
  static uint64_t data_id_gen_;
  uint64_t data_id_;
  DataNodeMeta meta_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataNode);
  void Init();
};

class OpNode: public DagNode {
 public:
  OpNode();
  ~OpNode();
  void set_runner(std::function<void()> r) { runner_ = r; }
  std::function<void()> runner() { return runner_; };
  NodeTypes Type() const { return OP_NODE; }

 private:
  std::function<void()> runner_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

} // end of namespace minerva
