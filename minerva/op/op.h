#pragma once

#include <string>

namespace minerva {

class BasicFn {
 public:
  virtual std::string Name() const = 0;
  virtual ~BasicFn() {}
};

class ClosureBase {
};

template<class T>
class ClosureTrait: public ClosureBase {
 public:
  T closure;
};

}
