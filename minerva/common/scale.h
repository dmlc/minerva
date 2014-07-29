#pragma once

#include <sstream>
#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>
#include <initializer_list>

namespace minerva {

class Scale;
class ScaleRange;

template<class T> class NVector;

class Scale {
  friend Scale operator + (const Scale& sc1, const Scale& sc2);
  friend Scale operator - (const Scale& sc1, const Scale& sc2);
  friend Scale operator * (const Scale& sc1, const Scale& sc2);
  friend Scale operator / (const Scale& sc1, const Scale& sc2);
  friend std::ostream& operator << (std::ostream& os, const Scale& sc);
 public:
  static const Scale kNullScale;
  static Scale Origin(size_t ndims) { return Scale(std::vector<int>(ndims, 0)); }
  static Scale Constant(size_t ndims, int val) { return Scale(std::vector<int>(ndims, val)); }

  Scale() {}
  Scale(const std::vector<int>& sc): vec_(sc) {} // allow implicit conversion
  Scale(const Scale& other): vec_(other.vec_) {}
  explicit Scale(int i1) {
    vec_.push_back(i1);
  }
  Scale(int i1, int i2) {
    vec_.push_back(i1);
    vec_.push_back(i2);
  }
  Scale(int i1, int i2, int i3) {
    vec_.push_back(i1);
    vec_.push_back(i2);
    vec_.push_back(i3);
  }
  int operator [] (size_t i) const { return vec_[i]; }
  int& operator [] (size_t i) { return vec_[i]; }
  bool operator == (const Scale& other) const {
    return vec_ == other.vec_;
  }
  bool operator != (const Scale& other) const {
    return vec_ != other.vec_;
  }
  Scale& operator = (const Scale& other) {
    vec_ = other.vec_;
    return *this;
  }
  bool operator < (const Scale& other) const {
    return vec_ < other.vec_;
  }
  bool operator <= (const Scale& other) const {
    return vec_ <= other.vec_;
  }
  bool operator > (const Scale& other) const {
    return vec_ > other.vec_;
  }
  bool operator >= (const Scale& other) const {
    return vec_ >= other.vec_;
  }
  size_t NumDims() const { return vec_.size(); }
  int Prod() const;
  std::string ToString() const;
  int get(int col) const {
    return vec_[col];
  }
  std::vector<int>::const_iterator begin() const {
    return vec_.begin();
  }
  std::vector<int>::const_iterator end() const {
    return vec_.end();
  }

  bool IncrOne(const Scale&);
  bool IncrWithOneDimensionFixed(const Scale&, size_t);
  NVector<Scale> EquallySplit(const Scale& numparts) const;
  static Scale Merge(const NVector<Scale>& partsizes);
  static bool IncrOne(Scale& pos, const Scale& max);
 private:
  std::vector<int> vec_;
};

inline std::ostream& operator << (std::ostream& os, const Scale& sc) {
  return os << sc.ToString();
}

class ScaleRange {
  friend std::ostream& operator << (std::ostream& os, const ScaleRange& range);
 public:
  static const ScaleRange kNullRange;
  static bool ValidRange(const Scale& st, const Scale& ed) {
    bool valid = st.NumDims() == ed.NumDims();
    if(valid) {
      for(size_t i = 0; valid && i < st.NumDims(); ++i) {
        valid &= (st[i] <= ed[i]);
      }
    }
    return valid;
  }
  static ScaleRange MakeRange(const Scale& st, const Scale& ed) {
    return ValidRange(st, ed) ? ScaleRange(st, ed) : kNullRange;
  }
  static ScaleRange MakeRangeFromOrigin(const Scale& len) {
    return ScaleRange(Scale::Origin(len.NumDims()), len);
  }
  static ScaleRange Intersect(const ScaleRange& r1, const ScaleRange& r2) {
    if(r1.NumDims() != r2.NumDims()) return kNullRange;
    std::vector<int> new_st, new_ed;
    for(size_t i = 0; i < r1.NumDims(); ++i) {
      new_st.push_back(std::max(r1.start_[i], r2.start_[i]));
      new_ed.push_back(std::min(r1.end_[i], r2.end_[i]));
    }
    return MakeRange(Scale(new_st), Scale(new_ed));
  }
  ScaleRange() {}
  ScaleRange(const ScaleRange& other): start_(other.start_), end_(other.end_) {}
  ScaleRange operator = (const ScaleRange& other) {
    start_ = other.start_;
    end_ = other.end_;
    return *this;
  }
  bool operator == (const ScaleRange& other) const  {
    return start_ == other.start_ && end_ == other.end_;
  }
  bool operator != (const ScaleRange& other) const  {
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
  bool IsInRange(const Scale& sc) const {
    return sc.NumDims() == start_.NumDims() &&
      sc >= start_ && sc < end_;
  }
  size_t Flatten(const Scale& sc) const {
    assert(IsInRange(sc));
    Scale off = sc - start_;
    Scale interval = end_ - start_;
    size_t stride = 1;
    size_t ret = 0;
    for(size_t i = 0; i < off.NumDims(); ++i) {
      ret += off[i] * stride;
      stride *= interval[i];
    }
    return ret;
  }
  Scale Dim() const {
    return end_ - start_;
  }

 private:
  ScaleRange(const Scale& st, const Scale& ed): start_(st), end_(ed) { }
 private:
  Scale start_, end_;
};

inline std::ostream& operator << (std::ostream& os, const ScaleRange& range) {
  return os << "{" << range.start_ << ", " << range.end_ << "}";
}

}// end of namespace minerva
