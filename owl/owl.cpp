#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/args.hpp>
#include <boost/numpy.hpp>
#include <boost/numpy/ndarray.hpp>

#include <glog/logging.h>

#include <iostream>
using namespace std;

#include "minerva.h"

namespace m = minerva;
namespace bp = boost::python;
namespace np = boost::numpy;

namespace owl {

void Initialize(bp::list args) {
  int argc = bp::extract<int>(args.attr("__len__")());
  char** argv = new char*[argc];
  for (int i = 0; i < argc; i++) {
    argv[i] = bp::extract<char*>(args[i]);
  }
  m::MinervaSystem::Instance().Initialize(&argc, &argv);
}

uint64_t CreateCpuDevice() {
  return m::MinervaSystem::Instance().device_manager().CreateCpuDevice();
}
#ifdef HAS_CUDA
uint64_t CreateGpuDevice(int id) {
  return m::MinervaSystem::Instance().device_manager().CreateGpuDevice(id);
}
#endif
void SetDevice(uint64_t id) {
  m::MinervaSystem::Instance().current_device_id_ = id;
}

m::NArray Softmax(m::NArray m) {
  m::NArray maxval = m.Max(0);
  // NArray centered = m - maxval.Tile({m.Size(0), 1});
  m::NArray centered = m.NormArithmetic(maxval, m::ArithmeticType::kSub);
  m::NArray class_normalizer = m::Elewise::Ln(m::Elewise::Exp(centered).Sum(0)) + maxval;
  // return Elewise::Exp(m - class_normalizer.Tile({m.Size(0), 1}));
  return m::Elewise::Exp(m.NormArithmetic(class_normalizer, m::ArithmeticType::kSub));
}

m::Scale ToScale(const bp::list& l) {
  bp::stl_input_iterator<int> begin(l), end;
  return m::Scale(begin, end);
}

bp::list ToPythonList(const m::Scale& s) {
  bp::list l;
  for(int i : s) {
    l.append(i);
  }
  return l;
}

m::NArray ZerosWrapper(const bp::list& s) {
  return m::NArray::Zeros(ToScale(s));
}

m::NArray OnesWrapper(const bp::list& s) {
  return m::NArray::Ones(ToScale(s));
}

m::NArray RandnWrapper(const bp::list& s, float mean, float var) {
  return m::NArray::Randn(ToScale(s), mean, var);
}

m::NArray RandBernoulliWrapper(const bp::list& s, float p) {
  return m::NArray::RandBernoulli(ToScale(s), p);
}

m::NArray ReshapeWrapper(m::NArray narr, const bp::list& s) {
  return narr.Reshape(ToScale(s));
}

m::NArray MakeNArrayWrapper(const bp::list& s, bp::list& val) {
  std::vector<float> v = std::vector<float>(bp::stl_input_iterator<float>(val), bp::stl_input_iterator<float>());
  size_t length = bp::len(val);
  shared_ptr<float> data( new float[length], [] (float* ptr) { delete [] ptr; } );
  memcpy(data.get(), v.data(), sizeof(float) * length);
//  for(size_t i = 0; i < length; ++i) {
//    valptr.get()[i] = bp::extract<float>(val[i] * 1.0);
//  }
  return m::NArray::MakeNArray(ToScale(s), data);
}

m::NArray FromNPArrayWrapper(np::ndarray nparr) {
  CHECK(nparr.get_flags() & np::ndarray::C_CONTIGUOUS) << "MakeNArray needs c-contiguous numpy array";
  CHECK(np::equivalent(nparr.get_dtype(), np::dtype::get_builtin<float>())) << "MakeNArray needs float32 numpy array";
  int nd = nparr.get_nd();
  m::Scale shape = m::Scale::Origin(nd);
  for(int i = 0; i < nd; ++i) {
    shape[i] = nparr.shape(nd - 1 - i);
  }
  size_t length = shape.Prod();
  shared_ptr<float> data( new float[length], [] (float* ptr) { delete [] ptr; } );
  memcpy(data.get(), reinterpret_cast<float*>(nparr.get_data()), sizeof(float) * length);
  return m::NArray::MakeNArray(shape, data);
}

bp::list ShapeWrapper(m::NArray narr) {
  return ToPythonList(narr.Size());
}

bp::list NArrayToList(m::NArray narr) {
  bp::list l;
  std::shared_ptr<float> v = narr.Get();
  for(int i = 0; i < narr.Size().Prod(); ++i)
    l.append(v.get()[i]);
  return l;
}

m::NArray ConvForward(m::NArray src, m::NArray filter, m::NArray bias, m::ConvInfo info) {
  return m::Convolution::ConvForward(m::ImageBatch(src), m::Filter(filter), bias, info);
}

m::NArray ActivationForward(m::NArray src, m::ActivationAlgorithm algo) {
  return m::Convolution::ActivationForward(m::ImageBatch(src), algo);
}

m::NArray PoolingForward(m::NArray src, m::PoolingInfo info) {
  return m::Convolution::PoolingForward(m::ImageBatch(src), info);
}

m::NArray PoolingBackward(m::NArray diff, m::NArray top, m::NArray bottom, m::PoolingInfo info) {
  return m::Convolution::PoolingBackward(m::ImageBatch(diff), m::ImageBatch(top), m::ImageBatch(bottom), info);
}

m::NArray ConvBackwardData(m::NArray diff, m::NArray filter, m::ConvInfo info) {
  return m::Convolution::ConvBackwardData(m::ImageBatch(diff), m::Filter(filter), info);
}

m::NArray ConvBackwardFilter(m::NArray diff, m::NArray bottom, m::ConvInfo info) {
  return m::Convolution::ConvBackwardFilter(m::ImageBatch(diff), m::ImageBatch(bottom), info);
}

m::NArray ConvBackwardBias(m::NArray diff) {
  return m::Convolution::ConvBackwardBias(m::ImageBatch(diff));
}

m::NArray ActivationBackward(m::NArray diff, m::NArray top, m::NArray bottom, m::ActivationAlgorithm algo) {
  return m::Convolution::ActivationBackward(m::ImageBatch(diff), m::ImageBatch(top), m::ImageBatch(bottom), algo);
}

m::NArray SoftmaxForward(m::NArray src, m::SoftmaxAlgorithm algo) {
  return m::Convolution::SoftmaxForward(m::ImageBatch(src), algo);
}

m::NArray SoftmaxBackward(m::NArray diff, m::NArray top, m::SoftmaxAlgorithm algo) {
  return m::Convolution::SoftmaxBackward(m::ImageBatch(diff), m::ImageBatch(top), algo);
}

void PrintProfilerResult() {
  m::MinervaSystem::Instance().profiler().PrintResult();
}

void ResetProfilerResult() {
  m::MinervaSystem::Instance().profiler().Reset();
}

} // end of namespace owl

// python module
BOOST_PYTHON_MODULE(libowl) {
  using namespace boost::python;
  np::initialize();

  enum_<m::ArithmeticType>("arithmetic")
    .value("add", m::ArithmeticType::kAdd)
    .value("sub", m::ArithmeticType::kSub)
    .value("mul", m::ArithmeticType::kMult)
    .value("div", m::ArithmeticType::kDiv)
  ;

  float (m::NArray::*sum0)() const = &m::NArray::Sum;
  m::NArray (m::NArray::*sum1)(int) const = &m::NArray::Sum;
  m::NArray (m::NArray::*sum2)(const m::Scale&) const = &m::NArray::Sum;

  float (m::NArray::*max0)() const = &m::NArray::Max;
  m::NArray (m::NArray::*max1)(int) const = &m::NArray::Max;
  m::NArray (m::NArray::*max2)(const m::Scale&) const = &m::NArray::Max;

  class_<m::Scale>("_Scale");

  class_<m::NArray>("NArray")
    // element-wise
    .def(self + self)
    .def(self - self)
    .def(self / self)
    .def(float() + self)
    .def(float() - self)
    .def(float() * self)
    .def(float() / self)
    .def(self + float())
    .def(self - float())
    .def(self * float())
    .def(self / float())
    .def(self += self)
    .def(self -= self)
    .def(self *= self)
    .def(self /= self)
    .def(self += float())
    .def(self -= float())
    .def(self *= float())
    .def(self /= float())
    // matrix multiply
    .def(self * self)
    // reduction
    .def("sum", sum0)
    .def("sum", sum1)
    .def("sum", sum2)
    .def("max", max0)
    .def("max", max1)
    .def("max", max2)
    .def("max_index", &m::NArray::MaxIndex)
    .def("count_zero", &m::NArray::CountZero)
    // normalize
    .def("norm_arithmetic", &m::NArray::NormArithmetic)
    // misc
    .def("trans", &m::NArray::Trans)
    .def("tofile", &m::NArray::ToFile)
    .def("tolist", &owl::NArrayToList)
    .def("reshape", &owl::ReshapeWrapper)
    .def("wait_for_eval", &m::NArray::WaitForEval)
    .def("start_eval", &m::NArray::StartEval)
    .add_property("shape", &owl::ShapeWrapper)
  ;
/*
  // file loader
  class_<m::IFileLoader>("IFileLoader");
  class_<m::SimpleFileLoader, bases<m::IFileLoader>>("SimpleFileLoader");
  class_<m::FileFormat>("FileFormat")
    .def_readwrite("binary", &m::FileFormat::binary)
  ;
*/
  // creators
  def("zeros", &owl::ZerosWrapper);
  def("ones", &owl::OnesWrapper);
  def("randn", &owl::RandnWrapper);
  def("make_narray", &owl::MakeNArrayWrapper);
  def("randb", &owl::RandBernoulliWrapper);
  def("from_nparray", &owl::FromNPArrayWrapper);
  //def("toscale", &owl::ToScale);

  // system
  //def("to_list", &owl::NArrayToList);
  def("initialize", &owl::Initialize);
  def("create_cpu_device", &owl::CreateCpuDevice);
#ifdef HAS_CUDA
  def("create_gpu_device", &owl::CreateGpuDevice);
#endif
  def("set_device", &owl::SetDevice);
  def("print_profiler_result", &owl::PrintProfilerResult);
  def("reset_profiler_result", &owl::ResetProfilerResult);

  // elewise
  def("mult", &m::Elewise::Mult);
  def("sigmoid", &m::Elewise::SigmoidForward);
  def("exp", &m::Elewise::Exp);
  def("ln", &m::Elewise::Ln);
  def("sigm", &m::Elewise::SigmoidForward);
  def("sigm_back", &m::Elewise::SigmoidBackward);
  def("relu", &m::Elewise::ReluForward);
  def("relu_back", &m::Elewise::ReluBackward);
  def("tahn", &m::Elewise::TanhForward);
  def("tahn_back", &m::Elewise::TanhBackward);

  // utils
  def("softmax", &owl::Softmax);

  // convolution
  class_<m::ConvInfo>("ConvInfo")
    .def_readwrite("pad_height", &m::ConvInfo::pad_height)
    .def_readwrite("pad_width", &m::ConvInfo::pad_width)
    .def_readwrite("stride_vertical", &m::ConvInfo::stride_vertical)
    .def_readwrite("stride_horizontal", &m::ConvInfo::stride_horizontal)
    ;
  class_<m::PoolingInfo>("PoolingInfo")
    .def_readwrite("height", &m::PoolingInfo::height)
    .def_readwrite("width", &m::PoolingInfo::width)
    .def_readwrite("stride_vertical", &m::PoolingInfo::stride_vertical)
    .def_readwrite("stride_horizontal", &m::PoolingInfo::stride_horizontal)
    .def_readwrite("algorithm", &m::PoolingInfo::algorithm)
    ;
  enum_<m::ActivationAlgorithm>("activation_algo")
    .value("relu", m::ActivationAlgorithm::kRelu)
    .value("sigm", m::ActivationAlgorithm::kSigmoid)
    .value("tanh", m::ActivationAlgorithm::kTanh)
  ;
  enum_<m::SoftmaxAlgorithm>("softmax_algo")
    .value("instance", m::SoftmaxAlgorithm::kInstance)
    .value("channel", m::SoftmaxAlgorithm::kChannel)
  ;
  enum_<m::PoolingInfo::Algorithm>("pooling_algo")
    .value("max", m::PoolingInfo::Algorithm::kMax)
    .value("avg", m::PoolingInfo::Algorithm::kAverage)
  ;

  def("conv_forward", &owl::ConvForward);
  def("activation_forward", &owl::ActivationForward);
  def("softmax_forward", &owl::SoftmaxForward);
  def("pooling_forward", &owl::PoolingForward);
  def("pooling_backward", &owl::PoolingBackward);
  def("conv_backward_data", &owl::ConvBackwardData);
  def("conv_backward_filter", &owl::ConvBackwardFilter);
  def("conv_backward_bias", &owl::ConvBackwardBias);
  def("activation_backward", &owl::ActivationBackward);
  def("softmax_backward", &owl::SoftmaxBackward);
}

