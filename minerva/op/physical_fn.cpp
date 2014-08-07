#include "op/physical_fn.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

DataStore& dstore = MinervaSystem::Instance().data_store();

DataShard::DataShard(const PhysicalData& d): data_info_(d) {
}

DataShard::DataShard(const DataShard& other): data_info_(other.data_info_) {
}

// return data untransformed (NO memory copy)
float* DataShard::GetCpuData() {
  return dstore.GetData(data_info_.data_id, DataStore::CPU);
}

float* DataShard::GetGpuData() {
  return dstore.GetData(data_info_.data_id, DataStore::GPU);
}

// return data transformed (may incur memory copy !!!)
float* DataShard::GetTransformedCpuData() {
  // TODO
  return GetCpuData();
}

float* DataShard::GetTransformedGpuData() {
  // TODO
  return GetGpuData();
}

}

