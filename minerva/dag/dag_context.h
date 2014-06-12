#pragma once

namespace minerva {

struct Place {
  int procid;
  int device_type; // 0 is CPU, 1 is GPU
  int device_id; // which core or which GPU
  Place(): procid(0), device_type(0), device_id(0) {}
};

struct OpNodeContext {
  Place place;
  int impl_type; // -1 is dynamic, 0 is basic, 1 is MKL, 2 is CUDA
  OpNodeContext(): impl_type(0) {}
};

struct DataNodeContext {
  bool transpose;
  DataNodeContext(): transpose(false) {}
};

class GlobalContext {
 public:
  static void SetOpPlace(const Place& place) {
    current_place_ = place;
  }
  static Place GetOpPlace() {
    return current_place_;
  }
  static void SetOpImpl(int impl) {
    current_impl_ = impl;
  }
  static int GetOpImpl() {
    return current_impl_;
  }
 private:
  static Place current_place_;
  static int current_impl_;
};

} // end of namespace minerva
