#include "dag.h"
#include "dag_node.h"
#include "concurrent_blocking_queue.h"
#include <cstdint>
#include <functional>
#include <queue>
#include <cstdio>

using namespace std;

uint64_t Dag::indexCounter = 0;

Dag::Dag() {
}

Dag::~Dag() {
    for (auto i: indexToNode) {
        delete i.second;
    }
}

DataNode* Dag::NewDataNode() {
    DataNode* ret = new DataNode;
    ret->nodeID = indexCounter++;
    indexToNode.insert(pair<uint64_t, DagNode*>(ret->nodeID, ret));
    return ret;
}

OpNode* Dag::NewOpNode() {
    OpNode* ret = new OpNode;
    ret->nodeID = indexCounter++;
    indexToNode.insert(pair<uint64_t, DagNode*>(ret->nodeID, ret));
    return ret;
}

DagNode* Dag::Root() {
    return root;
}

void Dag::TraverseAndRun() {
    ConcurrentBlockingQueue<DagNode*> q;
    auto succ = root->successors;
    for (auto i: succ) {
        q.Push(i);
        i->DeleteParent(root);
    }
    while (!q.Empty()) {
        DagNode* cur;
        q.Pop(cur);
        cur->Runner()();
        succ = cur->successors;
        for (auto i: succ) {
            i->DeleteParent(cur);
            if (i->IsSource()) {
                q.Push(i);
            }
        }
    }
}

