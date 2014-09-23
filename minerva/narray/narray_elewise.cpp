#include "narray/narray_elewise.h"
#include "narray/narray.h"
#include "op/physical_op.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

// Helper functions
static NArray UnaryElewiseCompute(const NArray& narr, PhysicalComputeFn* op) {
  return NArray::ComputeOne({narr}, narr.Size(), op);
}

static NArray BinaryElewiseCompute(const NArray& lhs, const NArray& rhs, PhysicalComputeFn* op) {
  CHECK_EQ(lhs.Size(), rhs.Size()) << "size must match";
  return NArray::ComputeOne({lhs, rhs}, lhs.Size(), op);
}

static NArray ElewiseHelper(const NArray& narr, ElewiseType type) {
  ElewiseOp* elewise_op = new ElewiseOp();
  elewise_op->closure = {type};
  return UnaryElewiseCompute(narr, elewise_op);
}

static NArray ArithmeticHelper(const NArray& lhs, const NArray& rhs, ArithmeticType type) {
  ArithmeticOp* arith_op = new ArithmeticOp();
  arith_op->closure = {type};
  return BinaryElewiseCompute(lhs, rhs, arith_op);
}

static NArray ArithmeticConstHelper(const NArray& narr, float val, int side, ArithmeticType type) {
  ArithmeticConstOp* arith_const_op = new ArithmeticConstOp();
  arith_const_op->closure = {type, val, side};
  return UnaryElewiseCompute(narr, arith_const_op);
}

// Element-wise operations
NArray Elewise::Mult(const NArray& lhs, const NArray& rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kMult);
}

NArray Elewise::Exp(const NArray& narr) {
  return ElewiseHelper(narr, ElewiseType::kExp);
}

NArray Elewise::Ln(const NArray& narr) {
  return ElewiseHelper(narr, ElewiseType::kLn);
}

NArray Elewise::Sigmoid(const NArray& narr) {
  return ElewiseHelper(narr, ElewiseType::kSigmoid);
}

NArray operator+(const NArray& lhs, NArray rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kAdd);
}

NArray operator-(const NArray& lhs, NArray rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kSub);
}

NArray operator/(const NArray& lhs, NArray rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kDiv);
}

NArray operator+(float lhs, NArray rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, ArithmeticType::kAdd);
}

NArray operator-(float lhs, NArray rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, ArithmeticType::kSub);
}

NArray operator*(float lhs, NArray rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, ArithmeticType::kMult);
}

NArray operator/(float lhs, NArray rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, ArithmeticType::kDiv);
}

NArray operator+(const NArray& lhs, float rhs) {
  return ArithmeticConstHelper(lhs, rhs, 1, ArithmeticType::kAdd);
}

NArray operator-(const NArray& lhs, float rhs) {
  return ArithmeticConstHelper(lhs, rhs, 1, ArithmeticType::kSub);
}

NArray operator*(const NArray& lhs, float rhs) {
  return ArithmeticConstHelper(lhs, rhs, 1, ArithmeticType::kMult);
}

NArray operator/(const NArray& lhs, float rhs) {
  return ArithmeticConstHelper(lhs, rhs, 1, ArithmeticType::kDiv);
}

void NArray::operator+=(const NArray& narr) {
  *this = ArithmeticHelper(*this, narr, ArithmeticType::kAdd);
}

void NArray::operator-=(const NArray& narr) {
  *this = ArithmeticHelper(*this, narr, ArithmeticType::kSub);
}

void NArray::operator*=(const NArray& narr) {
  *this = ArithmeticHelper(*this, narr, ArithmeticType::kMult);
}

void NArray::operator/=(const NArray& narr) {
  *this = ArithmeticHelper(*this, narr, ArithmeticType::kDiv);
}

void NArray::operator+=(float val) {
  *this = ArithmeticConstHelper(*this, val, 1, ArithmeticType::kAdd);
}

void NArray::operator-=(float val) {
  *this = ArithmeticConstHelper(*this, val, 1, ArithmeticType::kSub);
}

void NArray::operator*=(float val) {
  *this = ArithmeticConstHelper(*this, val, 1, ArithmeticType::kMult);
}

void NArray::operator/=(float val) {
  *this = ArithmeticConstHelper(*this, val, 1, ArithmeticType::kDiv);
}

NArray NArray::operator-() {
  return ElewiseHelper(*this, ElewiseType::kNegative);
}

} // end of namespace minerva

