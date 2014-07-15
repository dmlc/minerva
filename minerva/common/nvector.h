#pragma once

#include <functional>
#include "scale.h"

namespace minerva {

template<class T>
class NVector {
public:
	NVector() { }
	explicit NVector(const Scale& size) {
		data_.resize(size.Prod());
		range_ = ScaleRange::MakeRange(Scale::Origin(size.NumDims()), size);
	}
	explicit NVector(const ScaleRange& r): range_(r) {
		data_.resize(range_.Area());
	}
  NVector(const ScaleRange& r, const std::vector<T>& d): range_(r), data_(d) {}
	NVector(const NVector& other): data_(other.data_), range_(other.range_) {}
	const T& operator[] (const Scale& idx) const {
		return data_[range_.Flatten(idx)];
	}
	T& operator[] (const Scale& idx) {
		return data_[range_.Flatten(idx)];
	}
	size_t Length() const {
		return data_.size();
	}
	Scale Size() const {
		return range_.Dim();
	}
	int Size(int dim) const {
		return range_.Dim()[dim];
	}
	void Resize(const Scale& size) {
		data_.resize(size.Prod());
		range_ = ScaleRange::MakeRange(Scale::Origin(size.NumDims()), size);
	}
  template<class U, class Fn>
  NVector<U> Map(Fn fn) {
    std::vector<U> newdata;
    for(T d : data_) {
      newdata.push_back(fn(d));
    }
    return NVector<U>(range_, newdata);
  }
private:
	std::vector<T> data_;
	ScaleRange range_;
};

} // end of namespace minerva
