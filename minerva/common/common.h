#pragma once
#include <iostream>
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

}
