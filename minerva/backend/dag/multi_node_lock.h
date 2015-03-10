#pragma once
#include <list>
#include <mutex>
#include "common/common.h"
#include "dag/physical_dag.h"

namespace minerva {

class MultiNodeLock {
 public:
  MultiNodeLock() = default;
  template<typename T> MultiNodeLock(PhysicalDag*, const std::vector<T*>&);
  template<typename T> MultiNodeLock(PhysicalDag*, T*);
  DISALLOW_COPY_AND_ASSIGN(MultiNodeLock);
  ~MultiNodeLock() = default;

 private:
  std::list<std::lock_guard<std::mutex>> locks_;
};

template<typename T>
MultiNodeLock::MultiNodeLock(PhysicalDag* dag, const std::vector<T*>& nodes) {
  std::lock_guard<std::mutex> l(dag->m_);
  Iter(nodes, [this](PhysicalDataNode* node) {
    locks_.emplace_front(node->m_);
  });
}

template<typename T>
MultiNodeLock::MultiNodeLock(PhysicalDag* dag, T* node) {
  std::lock_guard<std::mutex> l(dag->m_);
  Iter(node->successors_, [this](DagNode* n) {
    locks_.emplace_front(n->m_);
  });
  Iter(node->predecessors_, [this](DagNode* n) {
    locks_.emplace_front(n->m_);
  });
  locks_.emplace_front(node->m_);
}

}  // namespace minerva

