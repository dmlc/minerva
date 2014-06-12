#pragma once

#include "Chunk.h"

namespace minerva {

struct Place {
	int procid;
	int type;
	int deviceid;
};

class PlaceContext {
public:
	static void SetOpPlace(const Place& place) {
		current_place_ = place;
	}
	static Place GetOpPlace() {
		return current_place_;
	}
private:
	static Place current_place_;
};

class DAGBuilder {
public:
	virtual std::vector<NVector<Chunk>> Compute(const std::vector<NVector<Chunk>>& inputs) = 0;
};


class MatMult : public DAGBuilder {
public:
	MatMult(const NVector<ChunkMeta>& l, const NVector<ChunkMeta>& r, const NVector<Place>& p):
		left_meta_(l), right_meta_(r), placement_(p) {
		//ASSERT(IsValid(l, r), "error message");
	}
	std::vector<NVector<Chunk>> Compute(const std::vector<NVector<Chunk>>& inputs) {
		NVector<Chunk> a = inputs[0];
		NVector<Chunk> b = inputs[1];
		//if(!IsValid(a, b)) {
			//Repartition(a, b);
		//}
		int m = a.Size()[0], n = b.Size()[1], k = a.Size()[1];
		NVector<Chunk> c({m, n});
		for(int i = 0; i < m; ++i) {
			for(int j = 0; j < n; ++j) {
				int row = a[{i, 0}].Size()[0];
				int col = b[{0, j}].Size()[1];
				c[{i, j}] = Chunk::Constant({row, col}, 0.0);
				PlaceContext::SetOpPlace(placement_[{i, j}]);
				for(int l = 0; l < k; ++l) {
					c[{i, j}] += a[{i, l}] * b[{l, j}];
				}
			}
		}
	}
	bool IsValid(const NVector<ChunkMeta>& a, const NVector<ChunkMeta>& b) {
		// TODO judge whether the input partition is the same with preset partitions
		return true;
	}
	void Repartition(const NVector<Chunk>& a, const NVector<Chunk>& b) {
		// TODO partition A and B according to the preset partition plan
	}
private:
	NVector<ChunkMeta> left_meta_, right_meta_; // preset partition plan
	NVector<Place> placement_;					// preset placement
};

} // end of namespace minerva

/*void TemporalMatMult() {
	Chunk w(1000, 500), x(500, 128);
	// Temporal partition of the activation matrix
	NVector<Chunk> xs = x.Split({1, 4});
	// Compute y = w * x;
	NVector<Chunk> ys({1, 4});
	for(size_t j = 0; j < 4; ++j) { // temporal
		Place::SetOpPlace({j, CPU, 0});
		ys[{0, j}] = w * xs[{0, j}];
	}
}

void SpatialMatMult() {
	Chunk w(1000, 500), x(500, 128);
	// Spatial partition of the activation matrix
	NVector<Chunk> ws = w.Split({4, 2});
	NVector<Chunk> xs = x.Split({2, 1});
	// Compute y = w * x;
	NVector<Chunk> ys({4, 1});
	for(size_t i = 0; i < 4; ++i) {
		Place::SetOpPlace({i, CPU, 0});
		ys[{i, 0}] = Chunk(250, 128, 0.0f);
		for(size_t j = 0; j < 2; ++j) {
			ys[{i, 0}] += ws[{i, j}] * xs[{j, 0}];
		}
	}
}*/

/*Layer l1, l2;
SetPartition(l1, {2});
A.AttachTo(l1);
B.AttachTo(l2);
C = A * B;

class M {
public:
	M(NVector<ChunkMeta> m): meta_(m) {}
	void AttachTo(Layer l1) {
		meta_ = l1.GetMeta();
	}
	M operator * (M& other) {
		MatMult matmult(meta_, other.meta_);
		return M(matmult.Compute());
	}
private:
	NVector<ChunkMeta> meta_;
};*/
