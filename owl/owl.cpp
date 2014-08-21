// Copyright 2014 Project Athena

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/implicit.hpp>

#include "minerva.h"

namespace bp = boost::python;
namespace m = minerva;

namespace owl {

void Initialize(bp::list args) {
  int argc = bp::extract<int>(args.attr("__len__")());
  char** argv = new char*[argc];
  for (int i = 0; i < argc; i++) {
    argv[i] = bp::extract<char*>(args[i]);
  }
  m::MinervaSystem::Instance().Initialize(&argc, &argv);
}

std::string LogicalDag() {
  return m::MinervaSystem::Instance().logical_dag().PrintDag();
}

m::NArray Softmax(m::NArray m) {
  m::NArray maxval = m.Max(0);
  // NArray centered = m - maxval.Tile({m.Size(0), 1});
  m::NArray centered = m.NormArithmetic(maxval, m::SUB);
  m::NArray class_normalizer = m::Elewise::Ln(m::Elewise::Exp(centered).Sum(0)) + maxval;
  // return Elewise::Exp(m - class_normalizer.Tile({m.Size(0), 1}));
  return m::Elewise::Exp(m.NormArithmetic(class_normalizer, m::SUB));
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

m::NArray ZerosWrapper(const bp::list& s, const bp::list& np) {
  return m::NArray::Zeros(ToScale(s), ToScale(np));
}

m::NArray OnesWrapper(const bp::list& s, const bp::list& np) {
  return m::NArray::Ones(ToScale(s), ToScale(np));
}

m::NArray MakeNArrayWrapper(const bp::list& s, bp::list& val, const bp::list& np) {
  size_t length = bp::len(val);
  std::shared_ptr<float> valptr(new float[length]);
  for(size_t i = 0; i < length; ++i) {
    valptr.get()[i] = bp::extract<float>(val[i]);
  }
  return m::NArray::MakeNArray(ToScale(s), valptr, ToScale(np));
}

m::NArray LoadFromFileWrapper(const bp::list& s, const std::string& fname, m::IFileLoader* loader, const bp::list& np) {
  return m::NArray::LoadFromFile(ToScale(s), fname, loader, ToScale(np));
}

class OneFileMBLoaderWrapper : public m::OneFileMBLoader {
 public:
  OneFileMBLoaderWrapper(const std::string& fname, const bp::list& shape):
    OneFileMBLoader(fname, ToScale(shape)) {}
};

bp::list NArrayToList(m::NArray narr) {
  bp::list l;
  float* v = narr.Get();
  for(int i = 0; i < narr.Size().Prod(); ++i)
    l.append(v[i]);
  delete [] v;
  return l;
}

void WaitForEvalFinish() {
  m::MinervaSystem::Instance().WaitForEvalFinish();
}

}

// python module
BOOST_PYTHON_MODULE(libowl) {
  using namespace boost::python;

  m::NArray (m::NArray::*fp_sum1)(int ) = &m::NArray::Sum;
  m::NArray (m::NArray::*fp_max1)(int ) = &m::NArray::Max;
  m::NArray (m::NArray::*fp_maxidx)(int ) = &m::NArray::MaxIndex;

  enum_<m::ArithmeticType>("arithmetic")
    .value("add", m::ADD)
    .value("sub", m::SUB)
    .value("mul", m::MULT)
    .value("div", m::DIV)
  ;

  //class_<m::Scale>("Scale");
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
    // matrix multiply
    .def(self * self)
    // reduction
    .def("sum", fp_sum1)
    .def("max", fp_max1)
    .def("argmax", fp_maxidx)
    .def("count_zero", &m::NArray::CountZero)
    // normalize
    .def("normalize", &m::NArray::NormArithmetic)
    // misc
    .def("trans", &m::NArray::Trans)
    .def("to_file", &m::NArray::ToFile)
    .def("eval", &m::NArray::Eval)
    .def("eval_async", &m::NArray::EvalAsync)
  ;

  // file loader
  class_<m::IFileLoader>("IFileLoader");
  class_<m::SimpleFileLoader, bases<m::IFileLoader>>("SimpleFileLoader");
  class_<m::FileFormat>("FileFormat")
    .def_readwrite("binary", &m::FileFormat::binary)
  ;
  def("load_from_file", &owl::LoadFromFileWrapper);

  // mb loader
  class_<owl::OneFileMBLoaderWrapper>("MBLoader", init<std::string, bp::list>())
    .def("load_next", &owl::OneFileMBLoaderWrapper::LoadNext)
  ;

  // creators
  def("zeros", &owl::ZerosWrapper);
  def("ones", &owl::OnesWrapper);
  def("make_narray", &owl::MakeNArrayWrapper);
  //def("random_randn", &m::NArray::Randn);
  //def("zeros", &m::NArray::Zeros);
  //def("ones", &m::NArray::Ones);

  // system
  def("to_list", &owl::NArrayToList);
  def("initialize", &owl::Initialize);
  def("logical_dag", &owl::LogicalDag);
  def("wait_eval", &owl::WaitForEvalFinish);

  // elewise
  def("mult", &m::Elewise::Mult);
  def("sigmoid", &m::Elewise::Sigmoid);
  def("exp", &m::Elewise::Exp);
  def("ln", &m::Elewise::Ln);
  
  // utils
  def("softmax", &owl::Softmax);
}
