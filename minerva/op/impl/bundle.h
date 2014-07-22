#pragma once

#include "basic.h"

namespace minerva {

template<class BasicFn>
class BundleTempl {
 public:
  BundleTempl(BasicFn bfn): basic_fn_(bfn) {}
 private:
  BasicFn basic_fn_;
};

}
