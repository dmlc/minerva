#pragma once
#include "common.h"
#include <atomic>

class BoolFlag {
 public:
  BoolFlag(bool f): flag_(f) {}
  ~BoolFlag() {}
  bool Read() const { return flag_;}
  void Write(bool f) { flag_ = f; }

 private:
  DISALLOW_COPY_AND_ASSIGN(BoolFlag);
  BoolFlag(): flag_(false) {}
  std::atomic<bool> flag_;
};
