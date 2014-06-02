#pragma once
#include "common/common.h"
#include <cstdint>
#include <vector>
#include <initializer_list>
#include <functional>
#include <mutex>

#include "dag_context.h"

namespace minerva {

class DagNode {
  friend class Dag;
  friend class DagEngine;

 public:
  DagNode();
  ~DagNode();
  void AddParent(DagNode*);
  void AddParents(std::initializer_list<DagNode*>);

 public:
  // setters
  void set_context(const DagNodeContext& ctx) { context_ = ctx; }
  // getters
  uint64_t node_id() { return node_id_; };
  std::function<void()> runner() { return runner_; };

 protected:
  bool DeleteParent(DagNode*);
  std::mutex mutex_;
  uint64_t node_id_;
  std::vector<DagNode*> successors_;
  std::vector<DagNode*> predecessors_;
  std::function<void()> runner_;
  DagNodeContext context_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DagNode);
};

class DataNode: public DagNode {
 public:
  DataNode();
  ~DataNode();
 private:
  DISALLOW_COPY_AND_ASSIGN(DataNode);
};

class OpNode: public DagNode {
 public:
  OpNode();
  ~OpNode();
 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

} // end of namespace minerva
