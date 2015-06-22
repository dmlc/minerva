#pragma once
#include <vector>
#include <functional>
#include "narray/narray.h"

namespace minerva {
namespace algorithm {

using NArrayBinaryOperator =
  std::function<NArray(NArray const&, NArray const&)>;

std::vector<NArray> AllReduce(
  std::vector<std::vector<NArray>> const& inputs, NArrayBinaryOperator);

}  // namespace algorithm
}  // namespace minerva

