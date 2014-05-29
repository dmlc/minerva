#pragma once

#include <athena/dag/dag.h>

namespace minerva {

class DAGProcedure {
public:
	virtual void Process(Dag& dag) = 0;
};

} // end of namespace minerva
