#pragma once
#include <iostream>
#include <vector>
#include <set>

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&); \
  void operator=(const TypeName&)

namespace minerva {

template<class T>
std::ostream& operator << (std::ostream& os, const std::set<T>& s) {
  os << "{";
  for(const T& t : s) os << t << " ";
  return os << "}";
}

template<class T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for(const T& t : v) os << v << " ";
  return os << "]";
}

}
