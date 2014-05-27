#pragma once

#include "NVector.h"
#include "DAGNode.h"

namespace minerva {

struct ChunkMeta;
class Chunk;
class ChunkOp;

struct ChunkMeta {
	ChunkMeta(): length(0) {}
	ChunkMeta(const ChunkMeta& other):
		length(other.length), size(other.size),
		offset(other.offset), chunk_index(other.chunk_index) {}
	ChunkMeta(const Index& size, const Index& off, 
			const Index& chidx):
		size(size), offset(off), chunk_index(chidx) {
		length = size.Prod();
	}
	ChunkMeta(const Index& size): size(size) {
		length = size.Prod();
		offset = Index::Origin(size.NumDims());
		chunk_index = Index::Origin(size.NumDims());
	}
	size_t length;
	Index size, offset, chunk_index;
};

class Chunk {
	friend Chunk operator * (const Chunk& a, const Chunk& b);
	friend Chunk operator + (const Chunk& a, const Chunk& b);
	friend Chunk operator += (const Chunk& a, const Chunk& b);
public:
	static Chunk Constant(const Index& size, float val);
public:
	Chunk() {}
	Index Size() const { return meta_.size; }
    DataNode* GetDataNode() const {
        return dataNode;
    }
private:
	ChunkMeta meta_;
    DataNode* dataNode; // Set up in constructor
};

} // end of namespace minerva
