#include "dag_node.h"
#include <initializer_list>

using namespace std;

void DagNode::AddParent(DagNode* p) {
    p->successors.push_back(this);
    predecessors.push_back(p);
}

void DagNode::AddParents(initializer_list<DagNode*> list) {
    for (auto i: list) {
        AddParent(i);
    }
}

