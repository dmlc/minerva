#include "chunk.h"
#include "op/closure.h"
#include "op/physical_op.h"
#include "op/shared_op.h"

#include <glog/logging.h>

using namespace std;

namespace minerva {

Chunk UnaryElewiseCompute(Chunk narr, PhysicalComputeFn* op) {
  return Chunk::Compute({narr}, {narr.Size()}, op)[0];
}

Chunk BinaryElewiseCompute(Chunk lhs, Chunk rhs, PhysicalComputeFn* op) {
  CHECK_EQ(lhs.Size(), rhs.Size()) << "(binary elewise) dimension mismatch";
  return Chunk::Compute({lhs, rhs}, {lhs.Size()}, op)[0];
}

Chunk ElewiseHelper(Chunk ch, enum ElewiseType type) {
  ElewiseOp* elewise_op = new ElewiseOp;
  elewise_op->closure = {type};
  return UnaryElewiseCompute(ch, elewise_op);
}

Chunk ArithmeticHelper(Chunk lhs, Chunk rhs, enum ArithmeticType type) {
  ArithmeticOp* arith_op = new ArithmeticOp;
  arith_op->closure = {type};
  return BinaryElewiseCompute(lhs, rhs, arith_op);
}

Chunk ArithmeticConstHelper(Chunk narr, float val, int side, enum ArithmeticType type) {
  ArithmeticConstOp* arith_const_op = new ArithmeticConstOp;
  arith_const_op->closure = {type, val, side};
  return UnaryElewiseCompute(narr, arith_const_op);
}

// arithmetic
Chunk operator + (Chunk a, Chunk b) {
  return ArithmeticHelper(a, b, ADD);
}
Chunk operator - (Chunk a, Chunk b) {
  return ArithmeticHelper(a, b, SUB);
}
Chunk operator / (Chunk a, Chunk b) {
  return ArithmeticHelper(a, b, DIV);
}
Chunk ChunkElewise::Mult(Chunk a, Chunk b) {
  return ArithmeticHelper(a, b, MULT);
};
void Chunk::operator += (Chunk a) {
  *this = *this + a;
}
void Chunk::operator -= (Chunk a) {
  *this = *this - a;
}
void Chunk::operator /= (Chunk a) {
  *this = *this / a;
}

// arithmetic const
Chunk operator + (Chunk a, float f) {
  return ArithmeticConstHelper(a, f, 1, ADD);
}
Chunk operator - (Chunk a, float f) {
  return ArithmeticConstHelper(a, f, 1, ADD);
}
Chunk operator * (Chunk a, float f) {
  return ArithmeticConstHelper(a, f, 1, ADD);
}
Chunk operator / (Chunk a, float f) {
  return ArithmeticConstHelper(a, f, 1, ADD);
}

Chunk operator + (float f, Chunk a) {
  return ArithmeticConstHelper(a, f, 0, ADD);
}
Chunk operator - (float f, Chunk a) {
  return ArithmeticConstHelper(a, f, 0, ADD);
}
Chunk operator * (float f, Chunk a) {
  return ArithmeticConstHelper(a, f, 0, ADD);
}
Chunk operator / (float f, Chunk a) {
  return ArithmeticConstHelper(a, f, 0, ADD);
}
void Chunk::operator += (float f) {
  *this = *this + f;
}
void Chunk::operator -= (float f) {
  *this = *this - f;
}
void Chunk::operator *= (float f) {
  *this = *this * f;
}
void Chunk::operator /= (float f) {
  *this = *this / f;
}

// elewise
Chunk ChunkElewise::Exp(Chunk ch) {
  return ElewiseHelper(ch, EXP);
}
Chunk ChunkElewise::Ln(Chunk ch) {
  return ElewiseHelper(ch, LN);
}
Chunk ChunkElewise::Sigmoid(Chunk ch) {
  return ElewiseHelper(ch, SIGMOID);
}
Chunk Chunk::operator - () {
  return ElewiseHelper(*this, NEGATIVE);
}

} // end of namespace minerva
