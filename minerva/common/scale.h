#pragma once
#include <sstream>
#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>
#include <utility>
#include <initializer_list>

namespace minerva {

class ScaleRange;

class Scale {
  friend Scale operator+(const Scale& sc1, const Scale& sc2);
  friend Scale operator-(const Scale& sc1, const Scale& sc2);
  friend Scale operator*(const Scale& sc1, const Scale& sc2);
  friend Scale operator/(const Scale& sc1, const Scale& sc2);
  friend Scale operator+(const Scale& sc1, int );
  friend Scale operator-(const Scale& sc1, int );
  friend Scale operator*(const Scale& sc1, int );
  friend Scale operator/(const Scale& sc1, int );
  friend std::ostream& operator<<(std::ostream& os, const Scale& sc);

 public:
  static const Scale kNullScale;
  static Scale Origin(size_t ndims) {
    return Scale(std::vector<int>(ndims, 0));
  }
  static Scale Constant(size_t ndims, int val) {
    return Scale(std::vector<int>(ndims, val));
  }
  Scale() {}
  Scale(const std::initializer_list<int>& lst) : vec_(lst) {}
  Scale(const std::vector<int>& sc) : vec_(sc) {}
  Scale(std::vector<int>&& sc) : vec_(std::move(sc)) {}
  template<typename Iter> Scale(const Iter& begin, const Iter& end) {
    for (Iter it = begin; it != end; ++it) {
      vec_.push_back(*it);
    }
  }
  Scale(const Scale& other): vec_(other.vec_) {}
  Scale(Scale&& other): vec_(std::move(other.vec_)) {}
  Scale& operator=(const Scale& other) {
    if (this == &other) {
      return *this;
    }
    vec_ = other.vec_;
    return *this;
  }
  Scale& operator=(Scale&& other) {
    if (this == &other) {
      return *this;
    }
    vec_ = std::move(other.vec_);
    return *this;
  }
  int operator[](size_t i) const {
    return vec_[i];
  }
  int& operator[](size_t i) {
    return vec_[i];
  }
  bool operator==(const Scale& other) const {
    return vec_ == other.vec_;
  }
  bool operator!=(const Scale& other) const {
    return vec_ != other.vec_;
  }
  bool operator<(const Scale& other) const {
    return vec_ < other.vec_;
  }
  bool operator<=(const Scale& other) const {
    return vec_ <= other.vec_;
  }
  bool operator>(const Scale& other) const {
    return vec_ > other.vec_;
  }
  bool operator>=(const Scale& other) const {
    return vec_ >= other.vec_;
  }
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
    for (auto i : vec_) {
      if (a == i) {
        return true;
      }
    }
    return false;
  }
  bool IncrOne(const Scale&);
  bool IncrWithDimensionsFixed(const Scale&, const Scale&);
  bool IncrDimensions(const Scale&, const Scale&);
  size_t NumDims() const {
    return vec_.size();
  }
  int Prod() const {
    int prod = vec_.empty() ? 0 : 1;
    for (auto i : vec_) {
      prod *= i;
    }
    return prod;
  }
  void Resize(size_t n, int val) {
    vec_.resize(n, val);
  }
  template<typename Fn> Scale Map(Fn fn) const {
    Scale ret;
    for (auto i : vec_) {
      ret.vec_.push_back(fn(i));
    }
    return ret;
  }
  Scale Concat(int val) const;
  const std::vector<int>& ToVector() const { return vec_; }
  std::string ToString() const;

 protected:
  std::vector<int> vec_;
};

inline std::ostream& operator<<(std::ostream& os, const Scale& sc) {
  return os << sc.ToString();
}

class ScaleRange {
  friend std::ostream& operator<<(std::ostream& os, const ScaleRange& range);

 public:
  static const ScaleRange kNullRange;
  static bool ValidRange(const Scale& st, const Scale& ed);
  template<typename T1, typename T2>
  static ScaleRange MakeRange(T1&& st, T2&& ed) {
    return ValidRange(st, ed) ? ScaleRange(std::forward<T1>(st), std::forward<T2>(ed)) : kNullRange;
  }
  template<typename T>
  static ScaleRange MakeRangeFromOrigin(T&& len) {
    return ScaleRange(Scale::Origin(len.NumDims()), std::forward<T>(len));
  }
  static ScaleRange Intersect(const ScaleRange& r1, const ScaleRange& r2);

 public:
  ScaleRange() {}
  ScaleRange(const ScaleRange& other) : start_(other.start_), end_(other.end_) {}
  ScaleRange(ScaleRange&& other) : start_(std::move(other.start_)), end_(std::move(other.end_)) {}
  ScaleRange& operator=(const ScaleRange& other) {
    if (this == &other) {
      return *this;
    }
    start_ = other.start_;
    end_ = other.end_;
    return *this;
  }
  ScaleRange& operator=(ScaleRange&& other) {
    if (this == &other) {
      return *this;
    }
    start_ = std::move(other.start_);
    end_ = std::move(other.end_);
    return *this;
  }
  bool operator==(const ScaleRange& other) const  {
    return (start_ == other.start_) && (end_ == other.end_);
  }
  bool operator!=(const ScaleRange& other) const  {
    return !(*this == other);
  }
  size_t NumDims() const {
    return start_.NumDims();
  }
  bool IsInRange(const Scale& sc) const {
    return (sc.NumDims() == start_.NumDims()) &&
        (start_ <= sc && sc < end_);
  }
  Scale Dim() const {
    return end_ - start_;
  }
  size_t Area() const;
  size_t Flatten(const Scale& sc) const;
  Scale start() const {
    return start_;
  }
  Scale end() const {
    return end_;
  }

 private:
  ScaleRange(const Scale& st, const Scale& ed) : start_(st), end_(ed) {}
  ScaleRange(const Scale& st, Scale&& ed) : start_(st), end_(std::move(ed)) {}
  ScaleRange(Scale&& st, const Scale& ed) : start_(std::move(st)), end_(ed) {}
  ScaleRange(Scale&& st, Scale&& ed) : start_(std::move(st)), end_(std::move(ed)) {}
  Scale start_, end_;
};

inline std::ostream& operator<<(std::ostream& os, const ScaleRange& range) {
  return os << "{" << range.start_ << ", " << range.end_ << "}";
}

}  // end of namespace minerva

