#include <op/context.h>
#include <minerva.h>
#include <iostream>
#include <gtest/gtest.h>

#include <op/logical_fn.h>
#include <op/physical_fn.h>

using namespace std;
using namespace minerva;

TEST(PerfTest, LotsOfUnusedNArray) {
  vector<NArray> narrs;
  for(int i = 0; i < 1000; ++i) {
    narrs.push_back(NArray::Constant({10, 10}, i, {1, 1}));
  }
  for(int i = 0; i < 1000; ++i) {
    narrs[i] = narrs[i] * 100 + 1;
  }
  for(int i = 0; i < 1000; ++i) {
    //cout << "eval narry#" << i << endl;
    narrs[i].Eval();
  }
}

TEST(PerfTest, LongChain) {
  NArray a = NArray::Constant({10, 10}, 0.0, {1, 1});
  for(int i = 0; i < 5000; ++i) {
    a += 1;
  }
  a.Eval();
  /*float* val = a.Get();
  for(int i = 0; i < 100; ++i)
    EXPECT_FLOAT_EQ(val[i], 5000) << "wrong value at i=" << i;
  delete [] val;*/
}

class AddOneManyTimesOp: public LogicalComputeFn, PhysicalComputeFn {
 public:
  void Execute(DataList& inputs, DataList& outputs, const Context&) {
    float* src = inputs[0].GetCpuData();
    float* dst = outputs[0].GetCpuData();
    memcpy(dst, src, inputs[0].Size().Prod() * sizeof(float));
    for(int j = 0; j < 5000; ++j) {
      for(int i = 0; i < inputs[0].Size().Prod(); ++i) {
        dst[i] += 1;
      }
    }
  }
  vector<NVector<Chunk>> Expand(const vector<NVector<Chunk>>& inputs) {
    NVector<Chunk> rst = inputs[0].Map<Chunk>(
        [] (const Chunk& ch) {
          return Chunk::Compute({ch}, {ch.Size()}, new AddOneManyTimesOp)[0];
        }
      );
    return {rst};
  }
  std::string Name() const { return "+1:5000times"; }
};

TEST(PerfTest, LongChainInOne) {
  NArray a = NArray::Constant({10, 10}, 0.0, {1, 1});
  NArray b = NArray::Compute({a}, {a.Size()}, new AddOneManyTimesOp)[0];
  b.Eval();
  /*float* val = b.Get();
  for(int i = 0; i < 100; ++i)
    EXPECT_FLOAT_EQ(val[i], 5000) << "wrong value at i=" << i;
  delete [] val;*/
}
