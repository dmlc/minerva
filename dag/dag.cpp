#include "dag.h"
#include "dag_node.h"
#include "concurrent_blocking_queue.h"
#include <cstdint>
#include <functional>
#include <queue>
#include <cstdio>
#include <thread>
#include <chrono>

using namespace std;

namespace minerva {

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
    ret->node_id_ = indexCounter++;
    indexToNode.insert(pair<uint64_t, DagNode*>(ret->node_id(), ret));
    ++unresolvedCounter;
    return ret;
}

OpNode* Dag::NewOpNode() {
    OpNode* ret = new OpNode;
    ret->node_id_ = indexCounter++;
    indexToNode.insert(pair<uint64_t, DagNode*>(ret->node_id(), ret));
    ++unresolvedCounter;
    return ret;
}

DagNode* Dag::Root() {
    return root;
}

void Dag::Worker(ConcurrentBlockingQueue<DagNode*>* queue) {
    while (true) {
        DagNode* cur;
        bool exitNow = queue->Pop(cur);
        if (exitNow) {
            return;
        }
        cur->runner()();
        --(this->unresolvedCounter);
        auto succ = cur->successors_;
        for (auto i: succ) {
            if (i->DeleteParent(cur)) {
                queue->Push(i);
            }
        }
    }
}

void Dag::TraverseAndRun() {
    ConcurrentBlockingQueue<DagNode*> q;
    auto succ = root->successors_;
    for (auto i: succ) {
        q.Push(i);
        i->DeleteParent(root);
    }
    --unresolvedCounter;
    std::thread t1(&Dag::Worker, this, &q);
    std::thread t2(&Dag::Worker, this, &q);
    std::thread t3(&Dag::Worker, this, &q);
    while (unresolvedCounter.load()) {
        std::this_thread::yield();
    }
    q.SignalForKill();
    t1.join();
    t2.join();
    t3.join();
}

} // end of namespace minerva
