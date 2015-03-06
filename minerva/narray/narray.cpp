#include "narray/narray.h"
#include "op/physical_op.h"
#include "common/common.h"
#include "system/minerva_system.h"
#include <fstream>
#include <glog/logging.h>
#include <iomanip>

using namespace std;

namespace minerva {

// Static constructors
NArray NArray::Constant(const Scale& size, float val) {
  FillOp* fill_op = new FillOp();
  fill_op->closure = {val};
  return NArray::GenerateOne(size, fill_op);
}

NArray NArray::Randn(const Scale& size, float mu, float var) {
  RandnOp* randn_op = new RandnOp();
  randn_op->closure = {mu, var};
  return NArray::GenerateOne(size, randn_op);
}

NArray NArray::RandBernoulli(const Scale& size, float p) {
  CHECK_LE(p, 1);
  CHECK_LE(0, p);
  RandBernoulliOp* op = new RandBernoulliOp();
  op->closure = {p};
  return NArray::GenerateOne(size, op);
}

/*NArray NArray::LoadFromFile(const Scale& size, const string& fname, shared_ptr<IFileLoader> loader) {
  FileLoaderOp* loader_op = new FileLoaderOp();
  loader_op->closure = {fname, size, loader};
  return NArray::GenerateOne(size, loader_op);
}*/

NArray NArray::Zeros(const Scale& size) {
  return NArray::Constant(size, 0.0);
}

NArray NArray::Ones(const Scale& size) {
  return NArray::Constant(size, 1.0);
}

NArray NArray::MakeNArray(const Scale& size, shared_ptr<float> array) {
  ArrayLoaderOp* loader_op = new ArrayLoaderOp();
  loader_op->closure = {array};
  return NArray::GenerateOne(size, loader_op);
}

NArray NArray::PushGradAndPullWeight(const NArray & grad, const std::string & layer_name) {
  SyncWithPSOp * op = new SyncWithPSOp();
  op->closure = { layer_name };
  return NArray::ComputeOne({ grad }, grad.Size(), op);
}

NArray& NArray::Pull(const std::string & layer_name) {
  SyncWithPSOp * op = new SyncWithPSOp();
  op->closure = { layer_name };
  return *this = NArray::GenerateOne(this->Size(), op);
}

// DAG building operations
vector<NArray> NArray::Compute(
    const vector<NArray>& params,
    const vector<Scale>& result_sizes,
    ComputeFn* fn) {
  auto& ms = MinervaSystem::Instance();
  auto param_mdata = Map<MData*>(params, [](const NArray& a) { return a.data_; });
  auto result_mdata = ms.Create(param_mdata, result_sizes, fn);
  return Map<NArray>(result_mdata, [](MData* md) { return NArray(md); });
}

NArray NArray::ComputeOne(const vector<NArray>& params, const Scale& size, ComputeFn* fn) {
  return Compute(params, {size}, fn)[0];
}

NArray NArray::GenerateOne(const Scale& size, ComputeFn* fn) {
  return NArray::ComputeOne({}, size, fn);
}

// Constructors and destructors
NArray::NArray() : data_(nullptr) {}

NArray::NArray(const NArray& other) : data_(nullptr) {
  MinervaSystem::Instance().ShallowCopy(data_, other.data_);
}

NArray::NArray(NArray&& other) : data_(other.data_) {
  other.data_ = nullptr;
}

NArray& NArray::operator=(const NArray& other) {
  if (this == &other) {
    return *this;
  }
  MinervaSystem::Instance().ShallowCopy(data_, other.data_);
  return *this;
}

NArray& NArray::operator=(NArray&& other) {
  if (this == &other) {
    return *this;
  }
  if (data_ != nullptr) {
    MinervaSystem::Instance().Destroy(data_);
  }
  data_ = other.data_;
  other.data_ = nullptr;
  return *this;
}

NArray::~NArray() {
  if (data_ != nullptr && MinervaSystem::IsAlive()) {
    MinervaSystem::Instance().Destroy(data_);
  }
}

// Matmult
NArray operator*(const NArray& lhs, const NArray& rhs) {
  CHECK_EQ(lhs.Size().NumDims(), 2) << "eligible only for 2D";
  CHECK_EQ(rhs.Size().NumDims(), 2) << "eligible only for 2D";
  CHECK_EQ(lhs.Size(1), rhs.Size(0)) << "size must match";
  Scale newsize = {lhs.Size(0), rhs.Size(1)};
  MatMultOp* matmult_op = new MatMultOp();
  return NArray::ComputeOne({lhs, rhs}, newsize, matmult_op);
}

NArray Concat(const std::vector<NArray>& arrays, int catdim) {
	CHECK_GT(arrays[0].Size().NumDims(), catdim) << "can't concat on non-sense dim";
	CHECK_GT(arrays.size(), 1) << "Concat more than one narray";
	ConcatOp* op = new ConcatOp();
	op->closure = {catdim}; 
	int target_dim = 0;
	for(size_t i = 0; i < arrays.size(); i++)
		target_dim += arrays[i].Size()[catdim];
	std::vector<int> sizevec(arrays[0].Size().NumDims(), 0);
	for(size_t i = 0; i < sizevec.size(); i++) {
		if(i == (size_t)catdim)
			sizevec[i] = target_dim;
		else
			sizevec[i] = arrays[0].Size()[i]; 
	}
	return NArray::ComputeOne(arrays, Scale(sizevec), op);
}

NArray Slice(const NArray& src, int slice_dim, int st_off, int slice_count)
{
	CHECK_GT(src.Size().NumDims(), slice_dim) << "can't concat on non-sense dim";
	std::vector<int> sizevec(src.Size().NumDims(), 0);
	for(size_t i = 0; i < sizevec.size(); i++) {
		if(i == (size_t)slice_dim)
			sizevec[i] = slice_count;
		else
			sizevec[i] = src.Size()[i]; 
	}
	SliceOp* op = new SliceOp();
	op->closure = {slice_dim, st_off, slice_count};
	return NArray::ComputeOne({src}, Scale(sizevec), op);
}


NArray& NArray::operator*=(const NArray& rhs) {
  return *this = (*this * rhs);
}

// Shape
NArray NArray::Reshape(const Scale& dims) const {
  CHECK_EQ(Size().Prod(), dims.Prod()) << "dimension mismatch";
  return NArray::ComputeOne({*this}, dims, new ReshapeOp());
}

NArray NArray::Trans() const {
  CHECK_EQ(Size().NumDims(), 2) << "eligible only for 2D";
  Scale newsize = {Size(1), Size(0)};
  TransOp* trans_op = new TransOp();
  return NArray::ComputeOne({*this}, newsize, trans_op);
}

// Replicate matrix
NArray NArray::NormArithmetic(const NArray& rhs, ArithmeticType type) const {
  auto& lhs = *this;
  CHECK_EQ(lhs.Size().NumDims(), rhs.Size().NumDims()) << "#dimension mismatch";
  vector<int> dims_to_replicate;
  // Dimensions to replicate
  for (size_t i = 0; i < lhs.Size().NumDims(); ++i) {
    if (lhs.Size()[i] == rhs.Size()[i]) {
      continue;
    } else if (rhs.Size()[i] != 1) {
      LOG(FATAL) << "NormArithmetic cannot replicate a dimension that is not 1";
    } else {
      dims_to_replicate.push_back(i);
	}
  }
  CHECK_GT(dims_to_replicate.size(), 0) << "nothing to replicate";
  NormArithmeticOp* op = new NormArithmeticOp();
  op->closure.type = type;
  op->closure.dims_to_replicate = dims_to_replicate;
  return NArray::ComputeOne({lhs, rhs}, lhs.Size(), op);
}

// System
void NArray::WaitForEval() const {
  MinervaSystem::Instance().Issue(data_);
  MinervaSystem::Instance().Wait(data_);
}

void NArray::StartEval() const {
  MinervaSystem::Instance().Issue(data_);
}

shared_ptr<float> NArray::Get() const {
  WaitForEval();
  return MinervaSystem::Instance().GetValue(data_);
}

void NArray::ToStream(ostream& out, const FileFormat& format) const {
  shared_ptr<float> ptr = Get();
  float* value = ptr.get();
  if (format.binary) {
    out.write(reinterpret_cast<char*>(value), Size().Prod() * sizeof(float));
  } else {
    for (int i = 0; i < Size().Prod(); ++i) {
      if (i != 0 && i % 10 == 0)
        out << "\n";
      out << setprecision(4) << value[i] << "\t";
    }
  }
}

void NArray::ToFile(const std::string& filename, const FileFormat& format) const {
  ofstream fout(filename.c_str());
  ToStream(fout, format);
  fout.close();
}

NArray::NArray(MData* data) : data_(nullptr) {
  MinervaSystem::Instance().ShallowCopy(data_, data);
}

}  // namespace minerva
