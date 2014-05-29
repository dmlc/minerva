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

void Dag::Worker(ConcurrentBlockingQueue<DagNode*>* queue) {
    while (true) {
        DagNode* cur;
        bool exitNow = queue->Pop(cur);
        if (exitNow) {
            return;
        }
        cur->Runner()();
        auto succ = cur->successors;
        for (auto i: succ) {
            if (i->DeleteParent(cur)) {
                queue->Push(i);
            }
        }
    }
}

void Dag::TraverseAndRun() {
    ConcurrentBlockingQueue<DagNode*> q;
    auto succ = root->successors;
    for (auto i: succ) {
        q.Push(i);
        i->DeleteParent(root);
    }
    std::thread t1(&Dag::Worker, this, &q);
    std::thread t2(&Dag::Worker, this, &q);
    std::thread t3(&Dag::Worker, this, &q);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    q.SignalForKill();
    t1.join();
    t2.join();
    t3.join();
}

