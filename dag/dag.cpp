#include "dag.h"
#include "dag_node.h"
#include <cstdint>

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

