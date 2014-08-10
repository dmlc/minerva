// Copyright 2014 Project Athena

#include <boost/python.hpp>

#include "minerva.h"

namespace bp = boost::python;
namespace m = minerva;

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

BOOST_PYTHON_MODULE(libowl) {
  using namespace boost::python;

  class_<m::Scale>("Scale", init<int>())
    .def(init<int, int>())
    .def(init<int, int, int>())
  ;
  class_<m::NArray>("NArray")
    // element-wise
    .def(self + self)
    .def(self - self)
    .def(self * self)  // exception: matrix multiplication
    .def(self / self)
    .def(float() + self)
    .def(float() - self)
    .def(float() * self)
    .def(float() / self)
    .def(self + float())
    .def(self - float())
    .def(self * float())
    .def(self / float())
    .def("trans", &m::NArray::Trans)
    .def("to_file", &m::NArray::ToFile)
  ;
  class_<m::IFileLoader>("IFileLoader");
  class_<m::SimpleFileLoader, bases<m::IFileLoader>>("SimpleFileLoader");
  class_<m::FileFormat>("FileFormat")
    .def_readwrite("binary", &m::FileFormat::binary)
  ;
  //def("random_randn", &m::NArray::Randn); // TODO [by jermaine] Not compiling
  def("initialize", &Initialize);
  def("load_from_file", &m::NArray::LoadFromFile);
  def("logical_dag", &LogicalDag);
}
