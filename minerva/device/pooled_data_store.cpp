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
  ds.length = length;
  DoCreateData(&ds.ptr, length);
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
  if (length == 0) {
    return common::MakeUnique<TemporarySpaceHolder>(nullptr);
  } else {
    lock_guard<mutex> lck(access_mutex_);
    DLOG(INFO) << "create temporary data of length " << length;
    float * ptr = nullptr;
    DoCreateData((void**)&ptr, length);
    auto deletor = [this, length, ptr]() {
      this->FreeTemporarySpace(ptr, length);
    };
    return std::unique_ptr<TemporarySpaceHolder>(new TemporarySpaceHolder(ptr, length, deletor));
  }
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

void PooledDataStore::DoCreateData(void** ptr, size_t length) {
  auto&& find_free_space = free_space_.find(length);
  if (find_free_space != free_space_.end()) {
    // reuse
    *ptr = find_free_space->second.front();
    find_free_space->second.pop();
    if (find_free_space->second.size() == 0) {
      free_space_.erase(find_free_space);
    }
  } else {
    total_ += length;
    if (threshold_ < total_) {
      ReleaseFreeSpace();
    }
    CHECK_GE(threshold_,  total_) << "not enough space";
    *ptr = allocator_(length);
    //cout << "Allocate size=" << length << " total=" << total_ << endl;
  }
}

void PooledDataStore::FreeTemporarySpace(void* ptr, size_t length) {
  lock_guard<mutex> lck(access_mutex_);
  free_space_[length].push(ptr);
}

}  // namespace minerva

