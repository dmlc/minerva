#include "dag_node.h"
#include <initializer_list>
#include <cstdio>
#include <algorithm>

using namespace std;

void DagNode::DeleteParent(DagNode* p) {
    predecessors.erase(std::find(predecessors.begin(), predecessors.end(), p));
    p->successors.erase(std::find(p->successors.begin(), p->successors.end(), this));
}

DagNode::DagNode() {
}

DagNode::~DagNode() {
}

void DagNode::AddParent(DagNode* p) {
    p->successors.push_back(this);
    predecessors.push_back(p);
}

void DagNode::AddParents(initializer_list<DagNode*> list) {
    for (auto i: list) {
        AddParent(i);
    }
}

uint64_t DagNode::ID() {
    return nodeID;
}

function<void()> DagNode::Runner() {
    return runner;
}

bool DagNode::IsSource() {
    return predecessors.empty();
}

DataNode::DataNode() {
    runner = [this] () {
        printf("Node %llu: Data Node\n", (unsigned long long) nodeID);
    };
}

DataNode::~DataNode() {
}

OpNode::OpNode() {
    runner = [this] () {
        printf("Node %llu: Op Node\n", (unsigned long long) nodeID);
    };
}

OpNode::~OpNode() {
}

