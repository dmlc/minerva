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

}  // namespace athena


