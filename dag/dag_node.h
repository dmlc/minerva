#pragma once
#include <cstdint>
#include <vector>
#include <initializer_list>
#include <functional>
#include <mutex>

#include "dag_context.h"

namespace minerva {

class DagNode {
    friend class Dag;
protected:
    std::mutex mutex_;
    uint64_t node_id_;
    std::vector<DagNode*> successors_;
    std::vector<DagNode*> predecessors_;
    std::function<void()> runner_;
	DagNodeContext context_;

protected:
    bool DeleteParent(DagNode*);

public:
    DagNode();
    ~DagNode();
    DagNode(const DagNode&);
    DagNode& operator=(const DagNode&);
    void AddParent(DagNode*);
    void AddParents(std::initializer_list<DagNode*>);
public:
	// getters
    uint64_t node_id() { return node_id_; }
    std::function<void()> runner() { return runner_; }
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

} // end of namespace minerva
