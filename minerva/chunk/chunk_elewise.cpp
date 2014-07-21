#include "chunk.h"
#include "op/closure.h"
#include "op/physical_op.h"

using namespace std;

namespace minerva {

Chunk UnaryElewiseCompute(Chunk narr, PhysicalComputeFn* op) {
  // return Chunk::Compute({narr}, {narr.Size()}, op)[0];
  return Chunk();
}

Chunk BinaryElewiseCompute(Chunk lhs, Chunk rhs, PhysicalComputeFn* op) {
  // assert(lhs.Size() == rhs.Size());
  // return Chunk::Compute({lhs, rhs}, {lhs.Size()}, op)[0];
  return Chunk();
}

Chunk ArithmeticHelper(Chunk lhs, Chunk rhs, enum ArithmeticType type) {
  // ArithmeticOp* arith_op = new ArithmeticOp;
  // arith_op->closure = {type};
  // return BinaryElewiseCompute(lhs, rhs, arith_op);
  return Chunk();
}

Chunk ArithmeticConstHelper(Chunk narr, float val, int side, enum ArithmeticType type) {
  // ArithmeticConstOp* arith_const_op = new ArithmeticConstOp;
  // arith_const_op->closure = {type, val, side};
  // return UnaryElewiseCompute(narr, arith_const_op);
  return Chunk();
}

void Chunk::operator += (float val) {
  *this = ArithmeticConstHelper(*this, val, 1, ADD);
}
void Chunk::operator += (Chunk ch) {
  *this = ArithmeticHelper(*this, ch, ADD);
}

} // end of namespace minerva
