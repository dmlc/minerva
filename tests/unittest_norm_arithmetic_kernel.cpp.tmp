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

void Fill(float* arr, float val, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    arr[i] = val;
  }
}

void Fill(float* arr, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    arr[i] = i;
  }
}

TEST(NormArithmeticKernel, AddFirstDimension) {
  DataStore& dstore = MinervaSystem::Instance().data_store();
  Scale s1 = {4, 6};
  Scale s2 = {1, 6};
  Scale s3 = s1;
  uint64_t id1 = dstore.GenerateDataId();
  uint64_t id2 = dstore.GenerateDataId();
  uint64_t id3 = dstore.GenerateDataId();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  dstore.CreateData(id3, DataStore::CPU, s3.Prod());
  PhysicalData d1 = MakeData(s1, id1);
  PhysicalData d2 = MakeData(s2, id2);
  PhysicalData d3 = MakeData(s3, id3);
  DataList in{DataShard(d1), DataShard(d2)};
  DataList out{DataShard(d3)};
  NormArithmeticClosure closure{ADD, Scale{0}};
  Fill(dstore.GetData(id1, DataStore::CPU), s1.Prod());
  Fill(dstore.GetData(id2, DataStore::CPU), 2, s2.Prod());
  basic::NormArithmetic(in, out, closure);
  float* res = dstore.GetData(id3, DataStore::CPU);
  for (int i = 0; i < s3[0]; ++i) {
    for (int j = 0; j < s3[1]; ++j) {
      EXPECT_FLOAT_EQ(res[i + s3[0] * j], i + s3[0] * j + 2);
    }
  }
}

TEST(NormArithmeticKernel, MultSecondDimension) {
  DataStore& dstore = MinervaSystem::Instance().data_store();
  Scale s1 = {4, 6};
  Scale s2 = {4, 1};
  Scale s3 = s1;
  uint64_t id1 = dstore.GenerateDataId();
  uint64_t id2 = dstore.GenerateDataId();
  uint64_t id3 = dstore.GenerateDataId();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  dstore.CreateData(id3, DataStore::CPU, s3.Prod());
  PhysicalData d1 = MakeData(s1, id1);
  PhysicalData d2 = MakeData(s2, id2);
  PhysicalData d3 = MakeData(s3, id3);
  DataList in{DataShard(d1), DataShard(d2)};
  DataList out{DataShard(d3)};
  NormArithmeticClosure closure{MULT, Scale{1}};
  Fill(dstore.GetData(id1, DataStore::CPU), s1.Prod());
  Fill(dstore.GetData(id2, DataStore::CPU), 2, s2.Prod());
  basic::NormArithmetic(in, out, closure);
  float* res = dstore.GetData(id3, DataStore::CPU);
  for (int i = 0; i < s3[0]; ++i) {
    for (int j = 0; j < s3[1]; ++j) {
      EXPECT_FLOAT_EQ(res[i + s3[0] * j], 2 * (i + s3[0] * j));
    }
  }
}

