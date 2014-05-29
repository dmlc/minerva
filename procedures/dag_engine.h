#pragma once

#include "dag_procedure.h"

namespace minerva {

class DAGEngine : public DAGProcedure {
public:
	void Process(Dag& dag) {
		// TODO execute the dag along the flow
	}
private:
	// TODO private members including but not limited to
	// 1. Threadpool
	// 2. Execution state (like counter)
};

}
