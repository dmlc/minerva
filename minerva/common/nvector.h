#pragma once

#include <functional>
#include "scale.h"

namespace minerva {

template<class T>
class NVector {
 public:
  // Constructor
  NVector() {}
  explicit NVector(const Scale& size) {
    data_.resize(size.Prod());
    range_ = ScaleRange::MakeRange(Scale::Origin(size.NumDims()), size);
  }
  explicit NVector(const ScaleRange& r): range_(r) {
    data_.resize(range_.Area());
  }
  NVector(const std::vector<T>& d, const ScaleRange& r): data_(d), range_(r) {}
  NVector(const std::vector<T>& d, const Scale& size): data_(d) {
    range_ = ScaleRange::MakeRangeFromOrigin(size);
  }
  NVector(const NVector& other): data_(other.data_), range_(other.range_) {}
  NVector(NVector&& other): data_(other.data_), range_(other.range_) {}
  NVector& operator = (const NVector& other) {
    data_ = other.data_;
    range_ = other.range_;
    return *this;
  }
  // Operator
  const T& operator[] (const Scale& idx) const {
    return data_[range_.Flatten(idx)];
  }
  T& operator[] (const Scale& idx) {
    return data_[range_.Flatten(idx)];
  }
  bool operator== (const NVector<T>& other) const {
    return data_ == other.data_ && range_ == other.range_;
  }
  // Utility function
  size_t Length() const {
    return data_.size();
  }
  Scale Size() const {
    return range_.Dim();
  }
  int Size(int dim) const {
    return range_.Dim()[dim];
  }
  bool IsEmpty() const {
  return range_.Area() == 0;
  }
  void Resize(const Scale& size) {
    data_.resize(size.Prod());
    range_ = ScaleRange::MakeRange(Scale::Origin(size.NumDims()), size);
  }
  const std::vector<T>& ToVector() const { return data_; }
  // Iterator and mapping
  template<class U, class Fn>
  NVector<U> Map(Fn fn) const {
    std::vector<U> newdata;
    for (const T& d: data_) {
      newdata.push_back(fn(d));
    }
    return NVector<U>(newdata, range_);
  }
  template<class Fn>
  static NVector<T> ZipMap(const NVector<T>& nv1, const NVector<T>& nv2, Fn fn) {
    assert(nv1.range_ == nv2.range_);
    std::vector<T> newdata;
    for (size_t i = 0; i < nv1.data_.size(); ++i) {
      newdata.push_back(fn(nv1.data_[i], nv2.data_[i]));
    }
    return NVector<T>(newdata, nv1.range_);
  }
  typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
  typename std::vector<T>::const_iterator end() const { return data_.end(); }
  T get(int i) const {
    return data_[i];
  }
  void AugmentDim() {
    range_ = ScaleRange::MakeRange(range_.start().Concat(1), range_.end().Concat(1));
  }

private:
  std::vector<T> data_;
  ScaleRange range_;
};

} // end of namespace minerva
