#pragma once
#include <cstdint>
#include <vector>
#include <initializer_list>
#include <functional>

class DagNode {
    friend class Dag;
protected:
    uint64_t nodeID;
    std::vector<DagNode*> successors;
    std::vector<DagNode*> predecessors;
    std::function<void()> runner;
public:
    DagNode();
    ~DagNode();
    DagNode(const DagNode&);
    DagNode& operator=(const DagNode&);
    void AddParent(DagNode*);
    void AddParents(std::initializer_list<DagNode*>);
    uint64_t ID();
    std::function<void()> Runner();
};

class DataNode: public DagNode {
public:
    DataNode();
    ~DataNode();
    DataNode(const DataNode&);
    DataNode& operator=(const DataNode&);
};

class OpNode: public DagNode {
public:
    OpNode();
    ~OpNode();
    OpNode(const OpNode&);
    OpNode& operator=(const OpNode&);
};

