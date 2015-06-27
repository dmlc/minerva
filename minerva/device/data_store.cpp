#include "./data_store.h"
#include "common/make_unique.h"

using namespace std;

namespace minerva {

DataStore::DataStore(function<void*(size_t)> a, function<void(void*)> d) : allocator_(a), deallocator_(d) {
}

DataStore::~DataStore() {
  for (auto& i : data_states_) {
    deallocator_(i.second.ptr);
  }
  CHECK_EQ(temporary_space_.size(), 0) << "temporary space not empty on exit";
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

std::unique_ptr<TemporarySpaceHolder>
DataStore::GetTemporarySpace(size_t length) {
  lock_guard<mutex> lck(access_mutex_);
  uint64_t id;
  // allocate new id
  if (temporary_space_.size() == 0) {
    id = 0;
  } else {
    id = temporary_space_.rbegin()->first + 1;
  }
  DLOG(INFO) << "create temporary data #" << id << " length " << length;
  auto&& it = temporary_space_.emplace(id, DataState());
  auto&& ds = it.first->second;
  ds.length = length;
  ds.ptr = allocator_(length);
  auto deallocator = [this, id]() {
    FreeTemporarySpace(id);
  };
  return
    common::MakeUnique<TemporarySpaceHolder>(ds.ptr, ds.length, deallocator);
}

void DataStore::FreeTemporarySpace(uint64_t id) {
  lock_guard<mutex> lck(access_mutex_);
  auto&& ds = temporary_space_.at(id);
  deallocator_(ds.ptr);
  CHECK_EQ(temporary_space_.erase(id), 1);
}

}  // namespace minerva

