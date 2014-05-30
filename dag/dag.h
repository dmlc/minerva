#pragma once

#include "dag_node.h"
#include "concurrent_blocking_queue.h"
#include <cstdint>
#include <map>
#include <functional>
#include <atomic>

namespace minerva {

class Dag {
private:
    static uint64_t indexCounter;
    std::map<uint64_t, DagNode*> indexToNode;
    DagNode* root = NewOpNode();
    std::atomic<size_t> unresolvedCounter{0};
	// TODO [Jermaine] please make the Dag class as a pure graph data structure
	// without any logic for execution. Contact me if you have any problem here
public:
    void Worker(ConcurrentBlockingQueue<DagNode*>*);
    Dag();
    ~Dag();
    Dag(const Dag&);
    Dag& operator=(const Dag&);
    DataNode* NewDataNode();
    OpNode* NewOpNode();
    DagNode* Root();
    void TraverseAndRun();
};

} // end of namespace minerva
