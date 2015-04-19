#pragma once
#include "minerva.h"

namespace athena {

uint64_t CreateCpuDevice();
uint64_t CreateGpuDevice(int);
int GetGpuDeviceCount();
void WaitForAll();
void SetDevice(uint64_t);

}  // namespace athena


