#pragma once

namespace minerva {

struct Place {
  int procid;
  int device_type; // 0 is CPU, 1 is GPU
  int device_id; // which core or which GPU
  Place(): procid(0), device_type(0), device_id(0) {}
};

class PlaceContext {
 public:
  static void SetOpContext(const Place& place) {
    current_place_ = place;
  }
  static Place GetOpContext() {
    return current_place_;
  }
 private:
  static Place current_place_;
};

struct DagNodeContext {
  Place place;
  int impl_type; // 0 is basic, 1 is MKL, 2 is CUDA
  DagNodeContext(): impl_type(0) {}
};

}
