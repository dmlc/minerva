#include "data_store.h"
#include <cstdlib>
#include <cstddef>
#include <glog/logging.h>

using namespace std;

namespace minerva {

DataStore::DataStore() {
}

DataStore::~DataStore() {
  for (auto& i: data_pointers_) {
    free(i.second);
  }
}

uint64_t DataStore::GenerateDataID() {
  static uint64_t data_id_gen = 0;
  return ++data_id_gen;
}

bool DataStore::CreateData(uint64_t id, MemTypes type, size_t size) {
  // TODO Allocate according to MemTypes
  std::lock_guard<std::mutex> lck(access_mutex_);
  LOG_IF(WARNING, data_pointers_.find(id) != data_pointers_.end())
    << "data_id(" << id << ") has already been created";
  auto ptr = data_pointers_.find(id);
  if (ptr != data_pointers_.end()) {
    FreeData(id, type); // Free existing storage
  }
  float* data = (float*) calloc(size, sizeof(float));
  data_pointers_[id] = data;
  return true;
}

float* DataStore::GetData(uint64_t id, MemTypes type) {
  std::lock_guard<std::mutex> lck(access_mutex_);
  auto ptr = data_pointers_.find(id);
  if (ptr == data_pointers_.end()) {
    LOG(WARNING) << "data_id(" << id << ") was not created!";
    return nullptr;
  }
  return ptr->second;
}

void DataStore::FreeData(uint64_t id, MemTypes type) {
  std::lock_guard<std::mutex> lck(access_mutex_);
  auto ptr = data_pointers_.find(id);
  if (ptr == data_pointers_.end()) {
    LOG(WARNING) << "data_id(" << id << ") was not created!";
    return;
  }
  free(ptr->second);
  data_pointers_.erase(ptr);
}

} // end of namespace minerva
