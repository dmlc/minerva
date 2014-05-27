#pragma once
#include <cstdint>
#include <vector>
#include <initializer_list>

class DagNode {
    friend class Dag;
private:
    uint64_t nodeID;
    std::vector<DagNode*> successors;
    std::vector<DagNode*> predecessors;
public:
    void AddParent(DagNode*);
    void AddParents(std::initializer_list<DagNode*>);
};

class DataNode: public DagNode {
};

class OpNode: public DagNode {
};

