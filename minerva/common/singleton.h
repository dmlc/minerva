#pragma once
#include <memory>
#include <glog/logging.h>

namespace minerva {
namespace common {

template<typename T>
class EverlastingSingleton {
 public:
  static T& Instance() {
    CHECK(data_) << "please initialize before use";
    return *data_;
  }
  static void Initialize(int* argc, char*** argv) {
    CHECK(!data_) << "already initialized";
    data_.reset(new T(argc, argv));
  }
  static void Finalize() {
    CHECK(data_) << "not alive";
    data_.release();
  }
  static bool IsAlive() {
    return static_cast<bool>(data_);
  }

 private:
  static std::unique_ptr<T> data_;
};

template<typename T> std::unique_ptr<T> EverlastingSingleton<T>::data_{};

}  // namespace common
}  // namespace minerva
