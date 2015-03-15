#include "data_store.h"

using namespace std;

namespace minerva {

DataStore::DataStore(function<void*(size_t)> a, function<void(void*)> d) : allocator_(a), deallocator_(d) {
}

DataStore::~DataStore() {
  for (auto& i : data_states_) {
    deallocator_(i.second.ptr);
  }
}

float* DataStore::CreateData(uint64_t id, size_t length) {
  lock_guard<mutex> lck(access_mutex_);
  DLOG(INFO) << "create data #" << id << " length " << length;
  auto it = data_states_.emplace(id, DataState());
  CHECK(it.second) << "data already existed";
  auto& ds = it.first->second;
  ds.length = length;
  ds.ptr = allocator_(length);
  return static_cast<float*>(ds.ptr);
}

float* DataStore::GetData(uint64_t id) {
  lock_guard<mutex> lck(access_mutex_);
  auto& ds = data_states_.at(id);
  return static_cast<float*>(ds.ptr);
}

bool DataStore::ExistData(uint64_t id) const {
  lock_guard<mutex> lck(access_mutex_);
  return data_states_.find(id) != data_states_.end();
}

void DataStore::FreeData(uint64_t id) {
  lock_guard<mutex> lck(access_mutex_);
  auto& ds = data_states_.at(id);
  deallocator_(ds.ptr);
  CHECK_EQ(data_states_.erase(id), 1);
}

size_t DataStore::GetTotalBytes() const {
  lock_guard<mutex> lck(access_mutex_);
  size_t total_bytes = 0;
  for (auto& it : data_states_) {
    total_bytes += it.second.length;
  }
  return total_bytes;
}

}  // namespace minerva
