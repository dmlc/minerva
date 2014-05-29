#include "bool_flag.h"
#include <atomic>

using namespace std;

BoolFlag::BoolFlag(): flag(false) {
}

BoolFlag::BoolFlag(bool f): flag(f) {
}

BoolFlag::~BoolFlag() {
}

bool BoolFlag::Read() const {
    return flag;
}

void BoolFlag::Write(bool f) {
    flag = f;
}

