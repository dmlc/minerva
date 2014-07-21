#include "chunk.h"
#include "op/closure.h"
#include "op/physical_op.h"

using namespace std;

namespace minerva {

Chunk UnaryElewiseCompute(Chunk narr, PhysicalComputeFn* op) {
  return Chunk::Compute({narr}, {narr.Size()}, op)[0];
}

Chunk BinaryElewiseCompute(Chunk lhs, Chunk rhs, PhysicalComputeFn* op) {
  assert(lhs.Size() == rhs.Size());
  return Chunk::Compute({lhs, rhs}, {lhs.Size()}, op)[0];
}

Chunk ArithmicHelper(Chunk lhs, Chunk rhs, enum ArithmicType type) {
  ArithmicOp* arith_op = new ArithmicOp;
  arith_op->closure = {type};
  return BinaryElewiseCompute(lhs, rhs, arith_op);
}

Chunk ArithmicConstHelper(Chunk narr, float val, int side, enum ArithmicType type) {
  ArithmicConstOp* arith_const_op = new ArithmicConstOp;
  arith_const_op->closure = {type, val, side};
  return UnaryElewiseCompute(narr, arith_const_op);
}

void Chunk::operator += (float val) {
  *this = ArithmicConstHelper(*this, val, 1, ADD);
}
void Chunk::operator += (Chunk ch) {
  *this = ArithmicHelper(*this, ch, ADD);
}

} // end of namespace minerva
