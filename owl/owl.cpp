// Copyright 2014 Project Athena

#include <boost/python.hpp>

#include "minerva.h"

namespace bp = boost::python;
namespace m = minerva;

std::string logical_dag() {
  return m::MinervaSystem::Instance().logical_dag().PrintDag();
}

BOOST_PYTHON_MODULE(libowl) {
  bp::class_<m::Scale>("Scale", bp::init<int>())
    .def(bp::init<int, int>())
    .def(bp::init<int, int, int>())
  ;
  bp::class_<m::NArray>("NArray")
    // element-wise
    .def(bp::self + bp::self)
    .def(bp::self - bp::self)
    .def(bp::self * bp::self)  // exception: matrix multiplication
    .def(bp::self / bp::self)
    .def(float() + bp::self)
    .def(float() - bp::self)
    .def(float() * bp::self)
    .def(float() / bp::self)
    .def(bp::self + float())
    .def(bp::self - float())
    .def(bp::self * float())
    .def(bp::self / float())
    .def("trans", &m::NArray::Trans)
  ;
  //bp::def("random_randn", &m::NArray::Randn); // TODO [by jermaine] Not compiling
  bp::def("logical_dag", &logical_dag);
}
