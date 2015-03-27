#include "narray/narray_elewise.h"
#include "narray/narray.h"
#include "op/physical_op.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

// Helper functions
static NArray ElewiseHelper(const NArray& narr, ElewiseType type) {
  ElewiseOp* elewise_op = new ElewiseOp();
  elewise_op->closure = {type};
  return NArray::ComputeOne({narr}, narr.Size(), elewise_op);
}

static NArray ArithmeticHelper(const NArray& lhs, const NArray& rhs, ArithmeticType type) {
  if (lhs.Size() == rhs.Size()) {
    CHECK_EQ(lhs.Size(), rhs.Size()) << "size must match";
    ArithmeticOp* arith_op = new ArithmeticOp();
    arith_op->closure = {type};
    return NArray::ComputeOne({lhs, rhs}, lhs.Size(), arith_op);
  } else {
    // Do NormArithmetic
    return lhs.NormArithmetic(rhs, type);
  }
}

static NArray ArithmeticConstHelper(const NArray& narr, float val, int side, ArithmeticType type) {
  ArithmeticConstOp* arith_const_op = new ArithmeticConstOp();
  arith_const_op->closure = {type, val, side};
  return NArray::ComputeOne({narr}, narr.Size(), arith_const_op);
}

// Element-wise operations
NArray Elewise::Mult(const NArray& lhs, const NArray& rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kMult);
}

NArray Elewise::Div(const NArray& lhs, const NArray& rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kDiv);
}

NArray Elewise::Exp(const NArray& narr) {
  return ElewiseHelper(narr, ElewiseType::kExp);
}

NArray Elewise::Ln(const NArray& narr) {
  return ElewiseHelper(narr, ElewiseType::kLn);
}

NArray Elewise::SigmoidForward(const NArray& narr) {
  return NArray::ComputeOne({narr}, narr.Size(), new SigmoidForwardOp());
}

NArray Elewise::SigmoidBackward(const NArray& diff, const NArray& top, const NArray& bottom) {
  CHECK_EQ(diff.Size(), top.Size()) << "inputs size mismatch";
  CHECK_EQ(diff.Size(), bottom.Size()) << "inputs size mismatch";
  return NArray::ComputeOne({diff, top, bottom}, diff.Size(), new SigmoidBackwardOp());
}

NArray Elewise::ReluForward(const NArray& narr) {
  return NArray::ComputeOne({narr}, narr.Size(), new ReluForwardOp());
}

NArray Elewise::ReluBackward(const NArray& diff, const NArray& top, const NArray& bottom) {
  CHECK_EQ(diff.Size(), top.Size()) << "inputs size mismatch";
  CHECK_EQ(diff.Size(), bottom.Size()) << "inputs size mismatch";
  return NArray::ComputeOne({diff, top, bottom}, diff.Size(), new ReluBackwardOp());
}

NArray Elewise::TanhForward(const NArray& narr) {
  return NArray::ComputeOne({narr}, narr.Size(), new TanhForwardOp());
}

NArray Elewise::TanhBackward(const NArray& diff, const NArray& top, const NArray& bottom) {
  CHECK_EQ(diff.Size(), top.Size()) << "inputs size mismatch";
  CHECK_EQ(diff.Size(), bottom.Size()) << "inputs size mismatch";
  return NArray::ComputeOne({diff, top, bottom}, diff.Size(), new TanhBackwardOp());
}

NArray operator+(const NArray& lhs, const NArray& rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kAdd);
}

NArray operator-(const NArray& lhs, const NArray& rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kSub);
}

NArray operator/(const NArray& lhs, const NArray& rhs) {
  return ArithmeticHelper(lhs, rhs, ArithmeticType::kDiv);
}

NArray operator+(float lhs, const NArray& rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, ArithmeticType::kAdd);
}

NArray operator-(float lhs, const NArray& rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, ArithmeticType::kSub);
}

NArray operator*(float lhs, const NArray& rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, ArithmeticType::kMult);
}

NArray operator/(float lhs, const NArray& rhs) {
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

NArray& NArray::operator+=(const NArray& narr) {
  return *this = (*this + narr);
}

NArray& NArray::operator-=(const NArray& narr) {
  return *this = (*this - narr);
}

NArray& NArray::operator/=(const NArray& narr) {
  return *this = (*this / narr);
}

NArray& NArray::operator+=(float val) {
  return *this = (*this + val);
}

NArray& NArray::operator-=(float val) {
  return *this = (*this - val);
}

NArray& NArray::operator*=(float val) {
  return *this = (*this * val);
}

NArray& NArray::operator/=(float val) {
  return *this = (*this / val);
}

NArray NArray::operator-() {
  return ElewiseHelper(*this, ElewiseType::kNegative);
}

} // end of namespace minerva

