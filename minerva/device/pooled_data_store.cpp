#include "device/pooled_data_store.h"

using namespace std;

namespace minerva {

PooledDataStore::PooledDataStore(size_t threshold, function<void*(size_t)> a, function<void(void*)> d) : DataStore(a, d), threshold_(threshold) {
}

PooledDataStore::~PooledDataStore() {
  for (auto& i : free_space_) {
    while (i.second.size()) {
      deallocator_(i.second.front());
      i.second.pop();
    }
  }
}

float* PooledDataStore::CreateData(uint64_t id, size_t length) {
  lock_guard<mutex> lck(access_mutex_);
  DLOG(INFO) << "create data id=" << id << "length=" << length;
  auto it = data_states_.emplace(id, DataState());
  CHECK(it.second) << "data already existed";
  auto& ds = it.first->second;
  ds.length = length;
  auto find_free_space_ = free_space_.find(length);
  if (find_free_space_ != free_space_.end()) {
    // Reuse
    ds.ptr = find_free_space_->second.front();
    find_free_space_->second.pop();
    if (!find_free_space_->second.size()) {
      free_space_.erase(find_free_space_);
    }
  } else {
    ds.ptr = allocator_(length);
    total_ += length;
    if (threshold_ < total_) {
      ReleaseFreeSpace();
    }
  }
  return static_cast<float*>(ds.ptr);
}

void PooledDataStore::FreeData(uint64_t id) {
  lock_guard<mutex> lck(access_mutex_);
  auto& ds = data_states_.at(id);
  free_space_[ds.length].push(ds.ptr);
  CHECK_EQ(data_states_.erase(id), 1);
}

size_t PooledDataStore::GetTotalBytes() const {
  lock_guard<mutex> lck(access_mutex_);
  return total_;
}

void PooledDataStore::ReleaseFreeSpace() {
  LOG(INFO) << "RELEASE";
  for (auto& i : free_space_) {
    while (i.second.size()) {
      deallocator_(i.second.front());
      i.second.pop();
      total_ -= i.first;
    }
  }
  free_space_.clear();
}

}  // namespace minerva

