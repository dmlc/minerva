#pragma once

namespace minerva {

struct Place {
	int procid;
	int device_type; // 0 is CPU, 1 is GPU
	int device_id; // which core or which GPU
	Place(): procid(0), device_type(0), device_id(0) {}
};

struct DagNodeContext {
	Place place;
	int impl_type; // 0 is basic, 1 is MKL, 2 is CUDA
	DagNodeContext(): impl_type(0) {}
};

}
