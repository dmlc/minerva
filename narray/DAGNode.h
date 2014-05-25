#pragma once
#include <vector>

namespace minerva {

class DagNode {
};

class OpNode: public DagNode {
public:
    enum OpType {
        mult
    };
    OpType type;
    std::vector<size_t> operands;
};

}
