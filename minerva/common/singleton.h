#pragma once
#include <glog/logging.h>

template<typename T>
class EverlastingSingleton {
 public:
  static T& Instance() {
    if (!data_) {
      CHECK(alive_);
      data_ = new T();
    }
    return *data_;
  }
  static bool IsAlive() {
    return alive_;
  }

 protected:
  void ShutDown() {
    alive_ = false;
    delete data_;
  }

 private:
  static T* data_;
  static bool alive_;
};

template<typename T> T* EverlastingSingleton<T>::data_ = 0;
template<typename T> bool EverlastingSingleton<T>::alive_ = true;

