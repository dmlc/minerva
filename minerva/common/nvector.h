#pragma once

#include "index.h"

namespace minerva {

template<class T>
class NVector {
public:
	NVector() { }
	NVector(const Index& size) {
		data_.resize(size.Prod());
		range_ = IndexRange::MakeRange(Index::Origin(size.NumDims()), size);
	}
	NVector(const IndexRange& r): range_(r) {
		data_.resize(range_.Area());
	}
	NVector(const NVector& other): data_(other.data_), range_(other.range_) {}
	const T& operator[] (const Index& idx) const {
		return data_[range_.Flatten(idx)];
	}
	T& operator[] (const Index& idx) {
		return data_[range_.Flatten(idx)];
	}
	size_t Length() const {
		return data_.size();
	}
	Index Size() const {
		return range_.Dim();
	}
	void Resize(const Index& size) {
		data_.resize(size.Prod());
		range_ = IndexRange::MakeRange(Index::Origin(size.NumDims()), size);
	}
private:
	std::vector<T> data_;
	IndexRange range_;
};

} // end of namespace minerva
