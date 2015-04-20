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

// Support enum class comparison for Cython
template<typename T>
int OfEvilEnumClass(T a) {
  return static_cast<int>(a);
}

template<typename T>
T ToEvilEnumClass(int a) {
  return static_cast<T>(a);
}

}  // namespace athena


