#include <op/context.h>
#include <minerva.h>
#include <iostream>
#include <gtest/gtest.h>

using namespace std;
using namespace minerva;

TEST(PerfTest, LotsOfUnusedNArray) {
  vector<NArray> narrs;
  for (int i = 0; i < 1000; ++i) {
    narrs.push_back(NArray::Constant({10, 10}, i));
  }
  for (int i = 0; i < 1000; ++i) {
    narrs[i] = narrs[i] * 100 + 1;
  }
  for (int i = 0; i < 1000; ++i) {
    narrs[i].WaitForEval();
  }
}

TEST(PerfTest, LongChain) {
  NArray a = NArray::Constant({10, 10}, 0.0);
  for (int i = 0; i < 5000; ++i) {
    a += 1;
  }
  a.WaitForEval();
}

class AddOneManyTimesOp : public PhysicalComputeFn {
 public:
  void Execute(const DataList& inputs, const DataList& outputs, const Context&) {
    float* src = inputs[0].data();
    float* dst = outputs[0].data();
    memcpy(dst, src, inputs[0].size().Prod() * sizeof(float));
    for (int j = 0; j < 5000; ++j) {
      for (int i = 0; i < inputs[0].size().Prod(); ++i) {
        dst[i] += 1;
      }
    }
  }
  std::string Name() const {
    return "+1:5000times";
  }
};

TEST(PerfTest, LongChainInOne) {
  NArray a = NArray::Constant({10, 10}, 0.0);
  NArray b = NArray::ComputeOne({a}, {a.Size()}, new AddOneManyTimesOp());
  b.WaitForEval();
}

