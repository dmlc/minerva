#pragma once

#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>

namespace minerva {

class Index;
class IndexRange;

class Index {
	friend Index operator + (const Index& idx1, const Index& idx2);
	friend Index operator - (const Index& idx1, const Index& idx2);
	friend Index operator * (const Index& idx1, const Index& idx2);
	friend std::ostream& operator << (std::ostream& os, const Index& idx);
 public:
	static const Index NullIndex;
	static Index Origin(size_t ndims) { return Index(std::vector<int>(ndims, 0)); }
	static Index Constant(size_t ndims, int val) { return Index(std::vector<int>(ndims, val)); }

	Index() {}
	Index(const std::vector<int>& idx): index_(idx) {} // allow implicit conversion
	Index(const std::vector<size_t>& idx): index_(idx.size()) {
		std::copy(idx.begin(), idx.end(), index_.begin());
	}
	Index(const Index& other): index_(other.index_) {}
	explicit Index(int i1) { // forbid implicit conversion
		index_.push_back(i1);
	}
	Index(int i1, int i2) {
		index_.push_back(i1);
		index_.push_back(i2);
	}
	Index(int i1, int i2, int i3) {
		index_.push_back(i1);
		index_.push_back(i2);
		index_.push_back(i3);
	}
	int operator [] (size_t i) const { return index_[i]; }
	int& operator [] (size_t i) { return index_[i]; }
	bool operator == (const Index& other) const {
		return index_ == other.index_;
	}
	bool operator != (const Index& other) const {
		return index_ != other.index_;
	}
	Index& operator = (const Index& other) {
		index_ = other.index_;
		return *this;
	}
	bool operator < (const Index& other) const {
		return index_ < other.index_;
	}
	bool operator <= (const Index& other) const {
		return index_ <= other.index_;
	}
	bool operator > (const Index& other) const {
		return index_ > other.index_;
	}
	bool operator >= (const Index& other) const {
		return index_ >= other.index_;
	}
	size_t NumDims() const { return index_.size(); }
	int Prod() const {
		if(index_.empty())
			return 0;
		else {
			int prod = 1;
			for(size_t i = 0; i < index_.size(); ++i) prod *= index_[i];
			return prod;
		}
	}
 private:
	std::vector<int> index_;
};

inline Index operator + (const Index& idx1, const Index& idx2) {
	assert(idx1.NumDims() == idx2.NumDims());
	std::vector<int> vec;
	for(size_t i = 0; i < idx1.NumDims(); ++i)
		vec.push_back(idx1[i] + idx2[i]);
	return Index(vec);
}
inline Index operator - (const Index& idx1, const Index& idx2) {
	assert(idx1.NumDims() == idx2.NumDims());
	std::vector<int> vec;
	for(size_t i = 0; i < idx1.NumDims(); ++i)
		vec.push_back(idx1[i] - idx2[i]);
	return Index(vec);
}
inline Index operator * (const Index& idx1, const Index& idx2) {
	assert(idx1.NumDims() == idx2.NumDims());
	std::vector<int> vec;
	for(size_t i = 0; i < idx1.NumDims(); ++i)
		vec.push_back(idx1[i] * idx2[i]);
	return Index(vec);
}
inline std::ostream& operator << (std::ostream& os, const Index& idx) {
	os << "[";
	for(size_t i = 0; i < idx.index_.size(); ++i)
		os << idx.index_[i] << " ";
	os << "]";
	return os;
}

class IndexRange {
	friend std::ostream& operator << (std::ostream& os, const IndexRange& range);
 public:
	static const IndexRange NullRange;
	static bool ValidRange(const Index& st, const Index& ed) {
		bool valid = st.NumDims() == ed.NumDims();
		if(valid) {
			for(size_t i = 0; valid && i < st.NumDims(); ++i) {
				valid &= (st[i] <= ed[i]);
			}
		}
		return valid;
	}
	static IndexRange MakeRange(const Index& st, const Index& ed) {
		return ValidRange(st, ed) ? IndexRange(st, ed) : NullRange;
	}
	static IndexRange Intersect(const IndexRange& r1, const IndexRange& r2) {
		if(r1.NumDims() != r2.NumDims()) return NullRange;
		std::vector<int> new_st, new_ed;
		for(size_t i = 0; i < r1.NumDims(); ++i) {
			new_st.push_back(std::max(r1.start_[i], r2.start_[i]));
			new_ed.push_back(std::min(r1.end_[i], r2.end_[i]));
		}
		return MakeRange(Index(new_st), Index(new_ed));
	}

	IndexRange() {}
	IndexRange(const IndexRange& other): start_(other.start_), end_(other.end_) {}
	IndexRange operator = (const IndexRange& other) {
		start_ = other.start_;
		end_ = other.end_;
		return *this;
	}
	bool operator == (const IndexRange& other) const  {
		return start_ == other.start_ && end_ == other.end_;
	}
	bool operator != (const IndexRange& other) const  {
		return ! (*this == other);
	}

	size_t NumDims() const { return start_.NumDims(); }
	size_t Area() const {
		size_t area = 1;
		bool all_zero = true;
		for(size_t i = 0; i < start_.NumDims(); ++i) {
			if(end_[i] != start_[i]) {
				area *= end_[i] - start_[i];
				all_zero = false;
			}
		}
		return all_zero ? 0 : area;
	}
	bool IsInRange(const Index& idx) const {
		return idx.NumDims() == start_.NumDims() && 
			idx >= start_ && idx < end_;
	}
	size_t Flatten(const Index& idx) const {
		assert(IsInRange(idx));
		Index off = idx - start_;
		Index interval = end_ - start_;
		size_t ret = 0;
		for(size_t i = 0; i < off.NumDims(); ++i) {
			ret = ret * interval[i] + off[i];
		}
		return ret;
	}
	Index Dim() const {
		return end_ - start_;
	}

 private:
	IndexRange(const Index& st, const Index& ed): start_(st), end_(ed) { }
 private:
	Index start_, end_;
};

inline std::ostream& operator << (std::ostream& os, const IndexRange& range) {
	return os << "{" << range.start_ << ", " << range.end_ << "}";
}

}// end of namespace minerva
