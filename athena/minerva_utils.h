#pragma once
#include "minerva.h"

namespace athena {

inline uint64_t CreateCpuDevice() {
  return minerva::MinervaSystem::Instance().device_manager().CreateCpuDevice();
}
}


