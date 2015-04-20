#pragma once
#include "minerva.h"
#include <vector>

namespace athena {

uint64_t CreateCpuDevice();
uint64_t CreateGpuDevice(int);
int GetGpuDeviceCount();
void WaitForAll();
void SetDevice(uint64_t);
minerva::Scale ToScale(std::vector<int>*);
std::vector<int> OfScale(minerva::Scale const&);

// struct PoolingAlgorithmWrapper {
//   using T = minerva::PoolingInfo::Algorithm;
//   struct W{
//     T a;
//   };
//   static constexpr W kMax
//     = {minerva::PoolingInfo::Algorithm::kMax};
//   static constexpr W kAverage
//     = {minerva::PoolingInfo::Algorithm::kAverage};
//   static constexpr T Extract(W w) {
//     return w.a;
//   }
// };

// Support enum class comparison for Cython
template<typename T>
int OfEvilEnumClass(T a) {
  return static_cast<int>(a);
}

template<typename T>
T ToEvilEnumClass(int a) {
  return static_cast<T>(a);
}

template<typename T>
bool EnumClassEqual(T a, T b) {
  return a == b;
}

}  // namespace athena


