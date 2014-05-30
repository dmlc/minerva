#pragma once
#include "common/common.h"
#include <cstdint>
#include <vector>
#include <initializer_list>
#include <functional>
#include <mutex>

class DagNode {
  friend class Dag;
 public:
  DagNode();
  ~DagNode();
  void AddParent(DagNode*);
  void AddParents(std::initializer_list<DagNode*>);
  uint64_t ID();
  std::function<void()> Runner();
 protected:
  bool DeleteParent(DagNode*);
  std::mutex mutex_;
  uint64_t node_id_;
  std::vector<DagNode*> successors_;
  std::vector<DagNode*> predecessors_;
  std::function<void()> runner_;
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

