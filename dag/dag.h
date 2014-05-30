#pragma once

#include "dag_node.h"
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include <cstdint>
#include <map>
#include <functional>
#include <atomic>

class Dag {
 public:
  Dag();
  ~Dag();
  DataNode* NewDataNode();
  OpNode* NewOpNode();

 private:
  DISALLOW_COPY_AND_ASSIGN(Dag);
  static uint64_t index_counter_;
  std::map<uint64_t, DagNode*> index_to_node_;
};

