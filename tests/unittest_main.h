#pragma once
#include <minerva.h>
#include <vector>
#include <cstdint>
#include <gtest/gtest.h>

extern uint64_t cpu_device;
#ifdef HAS_CUDA
extern uint64_t gpu_device;
extern std::vector<uint64_t> gpu_devices;
#endif

