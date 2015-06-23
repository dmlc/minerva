#include "narray/narray.h"
#include "op/physical_op.h"
#include <dmlc/logging.h>

using namespace std;

namespace minerva {

NArray NArray::SumAllExceptDim(const int dim) const {
	auto size = Size();
	//deprecated
	if(size.NumDims() == 2)
		return Sum(Scale{dim});
   	
	CHECK_LT(dims, size.NumDims()) << "reduce dim exceeds NArray dims";
	
	//Phase1: Reshape and reduction on Col
	NArray result_p1();
	if(dim > 0){
		ReductionWithReshapeOp* reductionwithreshape_op_p1 = new ReductionWithReshapeOp();
		reductionwithreshape_op_p2->closure.type = ReductionType::kSum;
		std::vector<int> reshape_vec_p1; 
		std::vector<int> newshapevec_p1(2, 1);
		for (int i = 0; i < size.NumDims(); i++){
			if(i < dim){
				newshapevec_p1[0] *= size[i];
				size[i] = 1;
			}
			else{
				newshapevec_p1[1] *= size[i];
			}
		}
		Scale newshape_p1(newshapevec_p1);
		reductionwithreshape_op_p1->newshape = newshape_p1;
		reductionwithreshape_op_p1->dims_to_reduce = 0;
		result_p1 = NArray::ComputeOne((*this), size, reductionwithreshape_op_p1);
	}
	else{
		result_p1 = (*this);
	}
	
	//Phase2: Reshape and reduction on Row
	if(dim < size.NumDims()){
		ReductionWithReshapeOp* reductionwithreshape_op_p2 = new ReductionWithReshapeOp();
		reductionwithreshape_op_p2->closure.type = ReductionType::kSum;
		std::vector<int> reshape_vec_p2; 
		std::vector<int> newshapevec_p2(2, 1);
		for (int i = dim; i < size.NumDims(); i++){
			if(i == dim){
				newshapevec_p2[0] = size[i];
			}
			else{
				newshapevec_p2[1] *= size[i];
				size[i] = 1;
			}
		}
		Scale newshape_p2(newshapevec_p2);
		reductionwithreshape_op->newshape = newshape_p2;
		reductionwithreshape_op->dims_to_reduce = 1;
		return NArray::ComputeOne(result_p1, size, reductionwithreshape_op);
	}
	else{
		return result_p1;
	}
}

	
// Lazy reductions
NArray NArray::Sum(int dim) const {
  return Sum(Scale{dim});
}

NArray NArray::Sum(const Scale& dims) const {
  CHECK_GT(dims.NumDims(), 0) << "nothing to reduce";
  auto size = Size();
  for (auto i : dims) {
    CHECK(0 <= i && i < static_cast<int>(size.NumDims())) << "dim out of bound";
    size[i] = 1;
  }
  ReductionOp* reduction_op = new ReductionOp();
  reduction_op->closure.type = ReductionType::kSum;
  reduction_op->closure.dims_to_reduce = dims;
  return NArray::ComputeOne({*this}, size, reduction_op);
}

NArray NArray::Max(int dim) const {
  return Max(Scale{dim});
}

NArray NArray::Max(const Scale& dims) const {
  CHECK_GT(dims.NumDims(), 0) << "nothing to reduce";
  auto size = Size();
  for (auto i : dims) {
    CHECK(0 <= i && i < static_cast<int>(size.NumDims())) << "dim out of bound";
    size[i] = 1;
  }
  ReductionOp* reduction_op = new ReductionOp();
  reduction_op->closure.type = ReductionType::kMax;
  reduction_op->closure.dims_to_reduce = dims;
  return NArray::ComputeOne({*this}, size, reduction_op);
}

NArray NArray::MaxIndex(int dim) const {
  auto size = Size();
  CHECK(0 <= dim && dim < static_cast<int>(size.NumDims())) << "dim out of bound";
  size[dim] = 1;
  MaxIndexOp* op = new MaxIndexOp();
  op->closure.dim = dim;
  return NArray::ComputeOne({*this}, size, op);
}

// Non-lazy reductions
float NArray::Sum() const {
  // TODO
  LOG(FATAL) << "not implemented";
  return 0;
}

float NArray::Max() const {
  // TODO
  LOG(FATAL) << "not implemented";
  return 0;
}

int NArray::CountZero() const {
  auto ptr = Get();
  auto value = ptr.get();
  int size = Size().Prod();
  int counter = 0;
  for (int i = 0; i < size; ++i) {
    if (!value[i]) {
      ++counter;
    }
  }
  return counter;
}

} // end of namespace minerva

