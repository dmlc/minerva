#pragma once

#include <string>

namespace minerva {

class BasicFn {
 public:
  virtual std::string Name() const = 0;
  virtual ~BasicFn() {}
};

class ClosureBase {
 public:
  virtual ~ClosureBase() {
  }
};

template<class T>
class ClosureTrait: public ClosureBase {
 public:
  T closure;
};

template<typename T>
T& GetClosureFromBase(ClosureBase* base) {
  return dynamic_cast<ClosureTrait<T>*>(base)->closure;
}

template<typename T>
ClosureBase* NewClosureBase(const T& closure) {
  auto trait = new ClosureTrait<T>;
  trait->closure = closure;
  return trait;
}

}

