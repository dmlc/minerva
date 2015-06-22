#include "./algorithm.h"
#include "common/common.h"
#include "system/minerva_system.h"

namespace minerva {
namespace algorithm {

std::vector<NArray> AllReduce(
    std::vector<std::vector<NArray>> const& inputs, NArrayBinaryOperator op) {
  auto&& ms = MinervaSystem::Instance();
  auto old_device_id = ms.current_device_id();
  auto device_count = inputs.size();
  CHECK_LT(0, device_count) << "empty input";
  auto single_device_num = inputs[0].size();
  CHECK_LT(0, single_device_num) << "empty input on single device";
  auto device_ids = Map<uint64_t>(
      inputs, [single_device_num](std::vector<NArray> const& input) {
    CHECK_EQ(single_device_num, input.size());
    auto device_id = input[0].GetDeviceId();
    for (auto&& i : input) {
      CHECK_EQ(device_id, i.GetDeviceId()) << "not on the same device";
    }
    return device_id;
  });
  std::vector<NArray> res;
  for (size_t i = 0; i < single_device_num; ++i) {
    res.emplace_back(inputs[i % device_count][i]);
  }
  // reduce
  for (size_t round = 1; round < device_count; ++round) {
    for (size_t i = 0; i < single_device_num; ++i) {
      auto target_id = (round + i) % device_count;
      ms.SetDevice(device_ids[target_id]);
      res[i] = op(res[i], inputs[target_id][i]);
    }
  }
  // scatter phase is automatic when the user gets the `NArray` from other
  // devices
  ms.SetDevice(old_device_id);
  return res;
}

}  // namespace algorithm
}  // namespace minerva
