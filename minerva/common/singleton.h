#pragma once

template<class T>
class Singleton {
 public:
  static T& Instance() {
    static T inst;
    return inst;
  }
};

template<class T>
class EverlastingSingleton {
 public:
  static T& Instance() {
    static T* inst_ptr = new T;
    return *inst_ptr;
  } 
};
