// Copyright 2014 Project Athena

#include <boost/python.hpp>

#include "common/index.h"
#include "chunk/chunk.h"

namespace bp = boost::python;
namespace m = minerva;

BOOST_PYTHON_MODULE(libowl) {
  bp::class_<m::Index>("Index")
    .def(bp::init<int, int>())
  ;
  bp::class_<m::Chunk>("Chunk")
    .def(bp::self * bp::self)
    .def("print_", &m::Chunk::Print)
  ;
  bp::def("constant_chunk", &m::Chunk::Constant);
}
