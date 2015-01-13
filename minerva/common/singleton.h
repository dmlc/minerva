#pragma once
#include <glog/logging.h>

template<typename T>
class EverlastingSingleton {
 public:
  static T& Instance() {
    CHECK(alive_) << "please initialize before use";
    return *data_;
  }
  static void Initialize(int* argc, char*** argv) {
    data_ = new T(argc, argv);
    alive_ = true;
  }
  static void Finalize() {
    delete data_;
    alive_ = false;
  }
  static bool IsAlive() {
    return alive_;
  }

 private:
  static T* data_;
  static bool alive_;
};

template<typename T> T* EverlastingSingleton<T>::data_ = 0;
template<typename T> bool EverlastingSingleton<T>::alive_ = false;

