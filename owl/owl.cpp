// Copyright 2014 Project Athena

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

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
  delete[] argv;
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

m::NArray ZerosWrapper(const bp::list& s, const bp::list& np) {
  bp::stl_input_iterator<int> sbegin(s), send;
  bp::stl_input_iterator<int> npbegin(np), npend;
  m::Scale size(sbegin, send);
  m::Scale numparts(npbegin, npend);
  return m::NArray::Zeros(size, numparts);
}

m::NArray OnesWrapper(const bp::list& s, const bp::list& np) {
  bp::stl_input_iterator<int> sbegin(s), send;
  bp::stl_input_iterator<int> npbegin(np), npend;
  m::Scale size(sbegin, send);
  m::Scale numparts(npbegin, npend);
  return m::NArray::Ones(size, numparts);
}

}

// python module
BOOST_PYTHON_MODULE(libowl) {
  using namespace boost::python;

  m::NArray (m::NArray::*fp_sum1)(int ) = &m::NArray::Sum;
  m::NArray (m::NArray::*fp_max1)(int ) = &m::NArray::Max;
  m::NArray (m::NArray::*fp_maxidx)(int ) = &m::NArray::MaxIndex;

  class_<m::Scale>("Scale")
    .def("numdims", &m::Scale::NumDims)
    .def("prod", &m::Scale::Prod)

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
  ;
  class_<m::IFileLoader>("IFileLoader");
  class_<m::SimpleFileLoader, bases<m::IFileLoader>>("SimpleFileLoader");
  class_<m::FileFormat>("FileFormat")
    .def_readwrite("binary", &m::FileFormat::binary)
  ;
  //def("random_randn", &m::NArray::Randn); // TODO [by jermaine] Not compiling
  def("zeros", &owl::ZerosWrapper);
  def("ones", &owl::OnesWrapper);

  def("initialize", &owl::Initialize);
  def("load_from_file", &owl::LoadFromFileWrapper);
  def("logical_dag", &owl::LogicalDag);

  // elewise
  def("mult", &m::Elewise::Mult);
  def("sigmoid", &m::Elewise::Sigmoid);
  def("exp", &m::Elewise::Exp);
  def("ln", &m::Elewise::Ln);
  
  // utils
  def("softmax", &owl::Softmax);
}
