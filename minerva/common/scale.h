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
  friend Scale operator + (const Scale& sc1, int );
  friend Scale operator - (const Scale& sc1, int );
  friend Scale operator * (const Scale& sc1, int );
  friend Scale operator / (const Scale& sc1, int );
  friend std::ostream& operator << (std::ostream& os, const Scale& sc);
 public:
  static const Scale kNullScale;
  static Scale Origin(size_t ndims) { return Scale(std::vector<int>(ndims, 0)); }
  static Scale Constant(size_t ndims, int val) { return Scale(std::vector<int>(ndims, val)); }

  // For iterator
  int get(int col) const {
    return vec_[col];
  }
  std::vector<int>::const_iterator begin() const {
    return vec_.begin();
  }
  std::vector<int>::const_iterator end() const {
    return vec_.end();
  }
  bool Contains(int a) const {
    for (auto i: vec_) {
      if (a == i) {
        return true;
      }
    }
    return false;
  }
  bool IncrOne(const Scale&);
  bool IncrWithDimensionsFixed(const Scale&, const Scale&);
  bool IncrDimensions(const Scale&, const Scale&);

  Scale() {}
  Scale(const std::initializer_list<int>& lst): vec_(lst) { Init(); } // allow implicit conversion
  Scale(const std::vector<int>& sc): vec_(sc) { Init(); } // allow implicit conversion
  Scale(const Scale& other): vec_(other.vec_), prod_val_(other.prod_val_) {}
  Scale(Scale&& other): vec_(other.vec_), prod_val_(other.prod_val_) {}
  Scale& operator = (const Scale& other) {
    vec_ = other.vec_;
    prod_val_ = other.prod_val_;
    return *this;
  }
  int operator [] (size_t i) const { return vec_[i]; }
  int& operator [] (size_t i) { return vec_[i]; }
  bool operator == (const Scale& other) const {
    return vec_ == other.vec_;
  }
  bool operator != (const Scale& other) const {
    return vec_ != other.vec_;
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
  int Prod() const { return prod_val_; }
  void Resize(size_t n, int val) { vec_.resize(n, val); }
  template<class Fn> Scale Map(Fn fn) const;
  Scale Concat(int val) const;

  NVector<Scale> EquallySplit(const Scale& numparts) const;
  static Scale Merge(const NVector<Scale>& partsizes);
  static bool IncrOne(Scale& pos, const Scale& max);

  const std::vector<int>& ToVector() const { return vec_; }
  std::string ToString() const;
  NVector<Scale> ToNVector() const;

 private:
  void Init() {
    prod_val_ = vec_.empty() ? 0 : 1;
    for(int i : vec_)
      prod_val_ *= i;
  }

 private:
  std::vector<int> vec_;
  int prod_val_;
};

inline std::ostream& operator << (std::ostream& os, const Scale& sc) {
  return os << sc.ToString();
}

template<class Fn>
Scale Scale::Map(Fn fn) const {
  Scale ret;
  for(size_t i = 0; i < vec_.size(); ++i) {
    ret.vec_.push_back(fn(vec_[i]));
  }
  return ret;
}

class ScaleRange {
  friend std::ostream& operator << (std::ostream& os, const ScaleRange& range);
 public:
  static const ScaleRange kNullRange;
  static bool ValidRange(const Scale& st, const Scale& ed);
  static ScaleRange MakeRange(const Scale& st, const Scale& ed) {
    return ValidRange(st, ed) ? ScaleRange(st, ed) : kNullRange;
  }
  static ScaleRange MakeRangeFromOrigin(const Scale& len) {
    return ScaleRange(Scale::Origin(len.NumDims()), len);
  }
  static ScaleRange Intersect(const ScaleRange& r1, const ScaleRange& r2);

 public:
  ScaleRange() {}
  ScaleRange(const ScaleRange& other): start_(other.start_), end_(other.end_) {}
  ScaleRange(ScaleRange&& other): start_(other.start_), end_(other.end_) {}
  ScaleRange& operator = (const ScaleRange& other) {
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
  bool IsInRange(const Scale& sc) const {
    return sc.NumDims() == start_.NumDims() &&
      sc >= start_ && sc < end_;
  }
  Scale Dim() const { return end_ - start_; }
  size_t Area() const;
  size_t Flatten(const Scale& sc) const;
  Scale start() const { return start_; }
  Scale end() const { return end_; }

 private:
  ScaleRange(const Scale& st, const Scale& ed): start_(st), end_(ed) { }
 private:
  Scale start_, end_;
};

inline std::ostream& operator << (std::ostream& os, const ScaleRange& range) {
  return os << "{" << range.start_ << ", " << range.end_ << "}";
}

extern void RangeCopy(float* src, const Scale& srcsize, const ScaleRange& srcrange,
    float* dst, const Scale& dstsize, const ScaleRange& dstrange);

}// end of namespace minerva
