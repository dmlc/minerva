#pragma once
#include <cstdint>

class DagNode {
private:
    uint64_t nodeID;
};

class DataNode: public DagNode {
};

class OpNode: public DagNode {
};

