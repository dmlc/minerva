#include "narray.h"
#include "op/logical_op.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

/////////////////////////////////////////////////////////////////////////
// Helper functions
NArray UnaryElewiseCompute(NArray narr, LogicalComputeFn* op) {
  return NArray::Compute({narr}, {narr.Size()}, op)[0];
}

NArray BinaryElewiseCompute(NArray lhs, NArray rhs, LogicalComputeFn* op) {
  assert(lhs.Size() == rhs.Size());
  return NArray::Compute({lhs, rhs}, {lhs.Size()}, op)[0];
}

NArray ElewiseHelper(NArray narr, enum ElewiseType type) {
  ElewiseOp* elewise_op = new ElewiseOp;
  elewise_op->closure = {type};
  return UnaryElewiseCompute(narr, elewise_op);
}

NArray ArithmeticHelper(NArray lhs, NArray rhs, enum ArithmeticType type) {
  ArithmeticOp* arith_op = new ArithmeticOp;
  arith_op->closure = {type};
  return BinaryElewiseCompute(lhs, rhs, arith_op);
}

NArray ArithmeticConstHelper(NArray narr, float val, int side, enum ArithmeticType type) {
  ArithmeticConstOp* arith_const_op = new ArithmeticConstOp;
  arith_const_op->closure = {type, val, side};
  return UnaryElewiseCompute(narr, arith_const_op);
}

/////////////////////////////////////////////////////////////////////////
// Definitions
NArray Elewise::Exp(NArray narr) {
  return ElewiseHelper(narr, EXP);
}
NArray Elewise::Ln(NArray narr) {
  return ElewiseHelper(narr, LN);
}
NArray Elewise::Sigmoid(NArray narr) {
  return ElewiseHelper(narr, SIGMOID);
}
NArray Elewise::Mult(NArray lhs, NArray rhs) {
  return ArithmeticHelper(lhs, rhs, MULT);
}
NArray NArray::operator - () {
  return ElewiseHelper(*this, NEGATIVE);
}
NArray operator + (NArray lhs, NArray rhs) {
  return ArithmeticHelper(lhs, rhs, ADD);
}
NArray operator - (NArray lhs, NArray rhs) {
  return ArithmeticHelper(lhs, rhs, SUB);
}
NArray operator / (NArray lhs, NArray rhs) {
  return ArithmeticHelper(lhs, rhs, DIV);
}
NArray operator + (float lhs, NArray rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, ADD);
}
NArray operator - (float lhs, NArray rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, SUB);
}
NArray operator * (float lhs, NArray rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, MULT);
}
NArray operator / (float lhs, NArray rhs) {
  return ArithmeticConstHelper(rhs, lhs, 0, DIV);
}
NArray operator + (NArray lhs, float rhs) {
  return ArithmeticConstHelper(lhs, rhs, 1, ADD);
}
NArray operator - (NArray lhs, float rhs) {
  return ArithmeticConstHelper(lhs, rhs, 1, SUB);
}
NArray operator * (NArray lhs, float rhs) {
  return ArithmeticConstHelper(lhs, rhs, 1, MULT);
}
NArray operator / (NArray lhs, float rhs) {
  return ArithmeticConstHelper(lhs, rhs, 1, DIV);
}

void NArray::operator += (NArray narr) {
  *this = ArithmeticHelper(*this, narr, ADD);
}
void NArray::operator -= (NArray narr) {
  *this = ArithmeticHelper(*this, narr, SUB);
}
void NArray::operator *= (NArray narr) {
  *this = ArithmeticHelper(*this, narr, MULT);
}
void NArray::operator /= (NArray narr) {
  *this = ArithmeticHelper(*this, narr, DIV);
}
void NArray::operator += (float val) {
  *this = ArithmeticConstHelper(*this, val, 1, ADD);
}
void NArray::operator -= (float val) {
  *this = ArithmeticConstHelper(*this, val, 1, SUB);
}
void NArray::operator *= (float val) {
  *this = ArithmeticConstHelper(*this, val, 1, MULT);
}
void NArray::operator /= (float val) {
  *this = ArithmeticConstHelper(*this, val, 1, DIV);
}

} // end of namespace minerva
