#pragma once
#include "common.h"
#include <atomic>

class BoolFlag {
 public:
  BoolFlag(bool);
  ~BoolFlag();
  bool Read() const;
  void Write(bool);

 private:
  DISALLOW_COPY_AND_ASSIGN(BoolFlag);
  BoolFlag();
  std::atomic<bool> flag_;
};

