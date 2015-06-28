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
  DLOG(INFO) << "create data #" << id << " length " << length;
  auto it = data_states_.emplace(id, DataState());
  CHECK(it.second) << "data already existed";
  auto&& ds = it.first->second;
  DoCreateData(&ds, length);
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

std::unique_ptr<TemporarySpaceHolder>
PooledDataStore::GetTemporarySpace(size_t length) {
  lock_guard<mutex> lck(access_mutex_);
  uint64_t id;
  // allocate new id
  if (temporary_space_.size() == 0) {
    id = 0;
  } else {
    id = temporary_space_.rbegin()->first + 1;
  }
  DLOG(INFO) << "create temporary data #" << id << " length " << length;
  auto&& it = temporary_space_.emplace(id, DataState{});
  auto&& ds = it.first->second;
  DoCreateData(&ds, length);
  auto deallocator = [this, id]() {
    FreeTemporarySpace(id);
  };
  return
    common::MakeUnique<TemporarySpaceHolder>(ds.ptr, ds.length, deallocator);
}

void PooledDataStore::ReleaseFreeSpace() {
  for (auto& i : free_space_) {
    while (i.second.size()) {
      deallocator_(i.second.front());
      i.second.pop();
      total_ -= i.first;
    }
  }
  free_space_.clear();
}

void PooledDataStore::DoCreateData(DataState* ds, size_t length) {
  ds->length = length;
  auto&& find_free_space = free_space_.find(length);
  if (find_free_space != free_space_.end()) {
    // reuse
    ds->ptr = find_free_space->second.front();
    find_free_space->second.pop();
    if (find_free_space->second.size() == 0) {
      free_space_.erase(find_free_space);
    }
  } else {
    ds->ptr = allocator_(length);
    total_ += length;
    if (threshold_ < total_) {
      ReleaseFreeSpace();
    }
  }
}

void PooledDataStore::FreeTemporarySpace(uint64_t id) {
  auto&& ds = temporary_space_.at(id);
  free_space_[ds.length].push(ds.ptr);
  CHECK_EQ(free_space_.erase(id), 1);
}

}  // namespace minerva

