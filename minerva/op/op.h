#pragma once

#include <string>

namespace minerva {

class BasicFn {
 public:
  virtual std::string Name() const = 0;
  virtual ~BasicFn() {}
};

template<class T>
class ClosureTrait {
 public:
  T closure;
};

} // end of namespace minerva
