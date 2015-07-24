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
  for (auto&& it : data_states_) {
    total_bytes += it.second.length;
  }
  // XXX temporary space is not counted
  return total_bytes;
}

std::unique_ptr<TemporarySpaceHolder>
DataStore::GetTemporarySpace(size_t length) {

  if (length == 0) {
    return common::MakeUnique<TemporarySpaceHolder>(nullptr);
  } else {
    lock_guard<mutex> lck(access_mutex_);
    DLOG(INFO) << "create temporary data of length " << length;
    float * ptr = static_cast<float*>(allocator_(length));
    auto deletor = [this, length, ptr]() {
      this->FreeTemporarySpace(ptr, length);
    };
    return std::unique_ptr<TemporarySpaceHolder>(new TemporarySpaceHolder(ptr, length, deletor));
  }
}

void DataStore::FreeTemporarySpace(void* ptr, size_t length) {
  deallocator_(ptr);
}

}  // namespace minerva

