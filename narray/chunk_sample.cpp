#include "Chunk.h"

using namespace std;
using namespace minerva;

struct Place {
	int procid;
	int type;
	int deviceid;
};

void TemporalMatMult() {
	Chunk w(1000, 500), x(500, 128);
	// Temporal partition of the activation matrix
	NDContainer<Chunk> xs = x.Split({1, 4});
	// Compute y = w * x;
	NDContainer<Chunk> ys({1, 4});
	for(size_t j = 0; j < 4; ++j) { // temporal
		Place::SetOpPlace({j, CPU, 0});
		ys[{0, j}] = w * xs[{0, j}];
	}
}

void SpatialMatMult() {
	Chunk w(1000, 500), x(500, 128);
	// Spatial partition of the activation matrix
	NDContainer<Chunk> ws = w.Split({4, 2});
	NDContainer<Chunk> xs = x.Split({2, 1});
	// Compute y = w * x;
	NDContainer<Chunk> ys({4, 1});
	for(size_t i = 0; i < 4; ++i) {
		Place::SetOpPlace({i, CPU, 0});
		ys[{i, 0}] = Chunk(250, 128, 0.0f);
		for(size_t j = 0; j < 2; ++j) {
			ys[{i, 0}] += ws[{i, j}] * xs[{j, 0}];
		}
	}
}

class DAGBuilder {
public:
	virtual vector<NDContainer<Chunk> > Compute(vector<NDContainer<Chunk> > inputs) = 0;
};

/*Layer l1, l2;
SetPartition(l1, {2});
A.AttachTo(l1);
B.AttachTo(l2);
C = A * B;

class M {
public:
	M(NDContainer<ChunkMeta> m): meta_(m) {}
	void AttachTo(Layer l1) {
		meta_ = l1.GetMeta();
	}
	M operator * (M& other) {
		MatMult matmult(meta_, other.meta_);
		return M(matmult.Compute());
	}
private:
	NDContainer<ChunkMeta> meta_;
};*/

class MatMult : public DAGBuilder {
public:
	MatMult(NDContainer<ChunkMeta> l, NDContaine<ChunkMeta> r) {
		ASSERT(IsValid(l, r), "error message");
	}
	vector<NDContainer<Chunk> > Compute(vector<NDContainer<Chunk> > inputs) {
		NDContainer<Chunk> A = inputs[0];
		NDContainer<Chunk> B = inputs[1];
		if(!IsValid(A, B)) {
			// repartition
		}
	}
	bool IsValid(A, B);
private:
	NDContainer<ChunkMeta> left_meta_, right_meta_;
	NDContainer<Place> left_place_, right_place_;
};
