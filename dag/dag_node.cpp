#include "dag_node.h"
#include <initializer_list>
#include <cstdio>
#include <algorithm>
#include <mutex>

using namespace std;

namespace minerva {

bool DagNode::DeleteParent(DagNode* p) {
    std::lock_guard<std::mutex> lock(mutex_);
    predecessors_.erase(std::find(predecessors_.begin(), predecessors_.end(), p));
    p->successors_.erase(std::find(p->successors_.begin(), p->successors_.end(), this));
    return predecessors_.empty();
}

DagNode::DagNode() {
}

DagNode::~DagNode() {
}

void DagNode::AddParent(DagNode* p) {
    std::lock_guard<std::mutex> lock(mutex_);
    p->successors_.push_back(this);
    predecessors_.push_back(p);
}

void DagNode::AddParents(initializer_list<DagNode*> list) {
    for (auto i: list) {
        AddParent(i);
    }
}

DataNode::DataNode() {
    runner_ = [this] () {
        printf("Node %llu: Data Node\n", (unsigned long long) node_id_);
    };
}

DataNode::~DataNode() {
}

OpNode::OpNode() {
    runner_ = [this] () {
        printf("Node %llu: Op Node\n", (unsigned long long) node_id_);
    };
}

OpNode::~OpNode() {
}

} // end of namespace minerva
