#pragma once
#include "Index.h"

namespace minerva {

struct ChunkMeta;
class Chunk;
class NMeta;
class NArray;

struct ConvInfo;
struct PoolingInfo;

struct ChunkMeta {
	ChunkMeta(): length(0) {}
	ChunkMeta(const utils::Index& dim, const utils::Index& off, 
			const utils::Index& offidx):
		size(dim), offset(off), chunk_index(offidx) {
		length = size.Prod();
	}
	ChunkMeta(const utils::Index& dim): size(dim) {
		length = size.Prod();
		offset = utils::Index::Origin(dim.size());
		chunk_index = utils::Index::Origin(dim.size());
	}
	size_t length;
	utils::Index size, offset, chunk_index;
};

class NMeta {
public:
	NMeta(const utils::NDContainer<ChunkMeta>& ml): meta_layout_(ml) {}
	const ChunkMeta& operator[] (const utils::Index& idx) const {
		return meta_layout_[idx];
	}
	ChunkMeta& operator[] (const utils::Index& idx) {
		return meta_layout_[idx];
	}
	NMeta SplitOnDim(size_t dimidx);
	size_t Size() const { return meta_layout_.Size(); }
	utils::NDContainer<ChunkMeta>& meta_layout() const { return meta_layout_; }
private:
	utils::NDContainer<ChunkMeta> meta_layout_;
};

class ChunkOp {
public:
	static Chunk ConvolveFF(const Chunk& input, const Chunk& filter, const ConvInfo& convinfo);
	static std::pair<Chunk, Chunk> ConvolveBP(const Chunk& input, const Chunk& filter, const ConvInfo& convinfo);
	static std::pair<Chunk, Chunk> PoolingFF(const Chunk& input, const PoolingInfo& poolinfo);
	static Chunk PoolingBP(const Chunk& input, const PoolingInfo& poolinfo);
};

class Chunk {
	friend Chunk operator + (const Chunk& lhs, const Chunk& rhs);
	friend Chunk operator * (const Chunk& lhs, const Chunk& rhs);
	friend Chunk operator - (const Chunk& lhs, const Chunk& rhs);

	friend Chunk operator + (float lhs, const Chunk& rhs);
	friend Chunk operator * (float lhs, const Chunk& rhs);
	friend Chunk operator - (float lhs, const Chunk& rhs);
	//friend Chunk operator / (float lhs, const Chunk& rhs);

	friend Chunk operator + (const Chunk& lhs, float rhs);
	friend Chunk operator * (const Chunk& lhs, float rhs);
	friend Chunk operator - (const Chunk& lhs, float rhs);
	friend Chunk operator / (const Chunk& lhs, float rhs);

	friend class ChunkOp;
public:
	Chunk operator - () const;
	Chunk& operator = (const Chunk& other);
	Chunk EleMult(const Chunk& rhs);
	Chunk EleDiv(const Chunk& rhs);
	Chunk Aggregate(size_t dim, OP_TYPE, float initval);
	Chunk Transpose();
	Chunk Reshape(const Index& new_size); // invarince: this->Size() == new_size.Prod();
	Chunk Repmat(const Index& rep_times);
	Chunk Slice(const Index& start, const Index& len);
	Chunk SliceOnDim(const vector<vector<size_t> >& idx_vector);
	Chunk Clone();
private:
	ChunkMeta meta_;
};

class NArrayOp {
public:
	static NArray ConvolveFF(const NArray& input, const NArray& filter, const ConvInfo& convinfo);
	static std::pair<NArray, NArray> ConvolveBP(const NArray& input, const NArray& filter, const ConvInfo& convinfo);
	static std::pair<NArray, NArray> PoolingFF(const NArray& input, const PoolingInfo& poolinfo);
	static NArray PoolingBP(const NArray& input, const PoolingInfo& poolinfo);
};

class NArray {
	friend NArray operator + (const NArray& lhs, const NArray& rhs);
	friend NArray operator * (const NArray& lhs, const NArray& rhs);
	friend NArray operator - (const NArray& lhs, const NArray& rhs);
	friend NArray operator / (const NArray& lhs, const NArray& rhs);

	friend NArray operator + (float lhs, const NArray& rhs);
	friend NArray operator * (float lhs, const NArray& rhs);
	friend NArray operator - (float lhs, const NArray& rhs);
	friend NArray operator / (float lhs, const NArray& rhs);

	friend NArray operator + (const NArray& lhs, float rhs);
	friend NArray operator * (const NArray& lhs, float rhs);
	friend NArray operator - (const NArray& lhs, float rhs);
	friend NArray operator / (const NArray& lhs, float rhs);

	friend class NArrayOp;
public:
	NArray operator - () const;
	NArray& operator = (const NArray& other);
	NArray EleMult(const NArray& rhs);
	NArray Aggregate(size_t dim, OP_TYPE);
	NArray Transpose();
	NArray Reshape(const Index& newdim);
	
	size_t NumChunks() const { return meta_.Size(); }
	size_t NumChunksOnDim(size_t dim) const { return meta_.meta_layout().Dim()[dim]; }
private:
	NMeta meta_;
};

int main() {
	Model model;
	Layer c1 = model.AddLayer(100);
	Layer c2 = model.AddLayer(1000);
	Layer c3 = model.AddLayer(500);
	Connection conn12 = model.AddConnection(c1, c2);
	Connection conn23 = model.AddConnection(c2, c3);
	model.SetPartition(c2, {2});
	model.SetPartition(c3, {2});
	model.Finalize();

	Matrix w21(c2, c1), w32(c3, c2);

	for(epoch = 0; epoch < MAX_EPOCH; ++epoch) {
		model.SetTemporalNumber(conn12, 16);
		// FF
		Matrix x(c1, 128);
		Matrix y = Sigmoid(w21 * x);
		Matrix z = Sigmoid(w32 * y);

		// BP
		Matrix t(c3, 128);
		Matrix dz = z - t;
		Matrix dy = w32.Trans() * dz;

		// Grad
		Matrix dw32 = dz * y.Trans();
		Matrix dw21 = dy * x.Trans();

		// Update
		w32 += dw32;
		w21 += dw21; // involve aggregation

		EvalAll();
	}

	return 0;
}

int main() {
	DAGPartitioner partitioner;
	DAGPlacement placement;

	MArray w(1000, 500), x(500, 128);

	{
	model.SetTemporal(wconn, 16);
	MArray y = w * x;
	}
	=>
	{
	MArray xs[16] = x.SplitOn(batchdim, 16);
	MArray ws[16] = w.Replicate(16);
	MArray ys[16];
	placement.SetPlacement(xs, {1, 2});
	placement.SetPlacement(ws, {1, 2});
	for(int i = 0; i < 16; ++i)
		ys[i] = ws[i] * xs[i];
	MArray y = Merge(ys);
	}

	{MArray z = w2 * y;}
	=>
	{
	partitioner.SetPartition(y, {4, 1});
	partitioner.SetPartition(w2, {5, 4});
	placement.SetPlacement(y, {1, 2});
	placement.SetPlacement(w2, {1, 2});
	MArray z = w2 * y;
	}

}




} // end of namespace minerva
