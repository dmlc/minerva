#include "bool_flag.h"
#include <atomic>

using namespace std;

BoolFlag::BoolFlag(): flag_(false) {
}

BoolFlag::BoolFlag(bool f): flag_(f) {
}

BoolFlag::~BoolFlag() {
}

bool BoolFlag::Read() const {
  return flag_;
}

void BoolFlag::Write(bool f) {
  flag_ = f;
}

