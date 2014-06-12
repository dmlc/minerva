#include "data_store.h"
#include <cstdlib>
#include <cstddef>

using namespace std;

namespace minerva {

DataStore::DataStore(): data_id_gen_(0) {
}

DataStore::~DataStore() {
  for (auto& i: data_pointers_) {
    free(i.second);
  }
}

uint64_t DataStore::GenerateDataID() {
  return data_id_gen_++;
}

bool DataStore::CreateData(uint64_t id, MemTypes type, size_t size) {
  // TODO Allocate according to MemTypes
  auto ptr = data_pointers_.find(id);
  if (ptr != data_pointers_.end()) {
    FreeData(id, type); // Free existing storage
  }
  float* data = (float*) calloc(size, sizeof(float));
  data_pointers_[id] = data;
  return true;
}

float* DataStore::GetData(uint64_t id, MemTypes type) {
  auto ptr = data_pointers_.find(id);
  if (ptr == data_pointers_.end()) {
    return nullptr;
  }
  return ptr->second;
}

void DataStore::FreeData(uint64_t id, MemTypes type) {
  auto ptr = data_pointers_.find(id);
  if (ptr == data_pointers_.end()) {
    return;
  }
  free(ptr->second);
  data_pointers_.erase(ptr);
}

} // end of namespace minerva

