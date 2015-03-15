#pragma once

#include <string>

namespace minerva {
  void PushGradAndPullWeight(const float * grad, float * weights, size_t size, const std::string & layer_name);
}