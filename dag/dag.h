#pragma once

#include "dag_node.h"
#include <cstdint>
#include <map>
#include <functional>

class Dag {
private:
    static uint64_t indexCounter;
    std::map<uint64_t, DagNode*> indexToNode;
    DagNode* root = NewOpNode();
public:
    Dag();
    ~Dag();
    Dag(const Dag&);
    Dag& operator=(const Dag&);
    DataNode* NewDataNode();
    OpNode* NewOpNode();
    DagNode* Root();
    void TraverseAndRun();
};

