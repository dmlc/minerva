#include <minerva.h>
#include <iostream>
#include <op/impl/basic.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

PhysicalData MakeData(Scale s, uint64_t data) {
  PhysicalData ret;
  ret.size = s;
  ret.offset = ret.offset_index = {0, 0};
  ret.data_id = data;
  return ret;
}

void Fill(float* arr, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    arr[i] = i;
  }
}

PhysicalData MakeData(Scale s, Scale o, Scale oi, uint64_t id) {
  PhysicalData pdata;
  pdata.size = s;
  pdata.offset = o;
  pdata.offset_index = oi;
  pdata.data_id = id;
  return pdata;
}

TEST(ReductionKernel, SumOnFirstDimension) {
  DataStore& dstore = MinervaSystem::Instance().data_store();
  Scale s1 = {20, 30};
  Scale s2 = {1, 30};
  uint64_t id1 = dstore.GenerateDataId();
  uint64_t id2 = dstore.GenerateDataId();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  PhysicalData d1 = MakeData(s1, id1);
  PhysicalData d2 = MakeData(s2, id2);
  DataList in{DataShard(d1)};
  DataList out{DataShard(d2)};
  ReductionClosure closure{SUM, Scale{0}};
  Fill(dstore.GetData(id1, DataStore::CPU), s1.Prod());
  basic::Reduction(in, out, closure);
  float* res = dstore.GetData(id2, DataStore::CPU);
  for (int i = 0; i < s2[1]; ++i) {
    EXPECT_FLOAT_EQ(res[i], 400 * i + 190);
  }
}

TEST(ReductionKernel, MaxOnSecondDimension) {
  DataStore& dstore = MinervaSystem::Instance().data_store();
  Scale s1 = {20, 30};
  Scale s2 = {20, 1};
  uint64_t id1 = dstore.GenerateDataId();
  uint64_t id2 = dstore.GenerateDataId();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  PhysicalData d1 = MakeData(s1, id1);
  PhysicalData d2 = MakeData(s2, id2);
  DataList in{DataShard(d1)};
  DataList out{DataShard(d2)};
  ReductionClosure closure{MAX, Scale{1}};
  Fill(dstore.GetData(id1, DataStore::CPU), s1.Prod());
  basic::Reduction(in, out, closure);
  float* res = dstore.GetData(id2, DataStore::CPU);
  for (int i = 0; i < s2[0]; ++i) {
    EXPECT_FLOAT_EQ(res[i], 580 + i);
  }
}

