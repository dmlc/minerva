#include <minerva.h>
#include <iostream>
#include <op/impl/basic.h>

using namespace minerva;
using namespace std;

void Fill(float* arr, float val, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    arr[i] = val;
  }
}

void Test1() {
  cout << "Test 2D reduction on first dimension" << endl;
  DataStore& dstore = MinervaSystem::Instance().data_store();
  Scale s1 = {20, 30};
  Scale s2 = {1, 30};
  uint64_t id1 = dstore.GenerateDataID();
  uint64_t id2 = dstore.GenerateDataID();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  PhysicalData d1{s1, {0, 0}, {0, 0}, id1, 0};
  PhysicalData d2{s2, {0, 0}, {0, 0}, id2, 0};
  DataList in{DataShard(d1)};
  DataList out{DataShard(d2)};
  ReductionClosure closure{SUM, Scale{0}};
  Fill(dstore.GetData(id1, DataStore::CPU), 1, s1.Prod());
  basic::Reduction(in, out, closure);
  float* res = dstore.GetData(id2, DataStore::CPU);
  for (int i = 0; i < s2[0]; ++i) {
    for (int j = 0; j < s2[1]; ++j) {
      cout << res[i * s2[1] + j] << " ";
    }
    cout << endl;
  }
}

void Test2() {
  cout << "Test 2D reduction on second dimension" << endl;
  DataStore& dstore = MinervaSystem::Instance().data_store();
  Scale s1 = {20, 30};
  Scale s2 = {20, 1};
  uint64_t id1 = dstore.GenerateDataID();
  uint64_t id2 = dstore.GenerateDataID();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  PhysicalData d1{s1, {0, 0}, {0, 0}, id1, 0};
  PhysicalData d2{s2, {0, 0}, {0, 0}, id2, 0};
  DataList in{DataShard(d1)};
  DataList out{DataShard(d2)};
  ReductionClosure closure{SUM, Scale{1}};
  Fill(dstore.GetData(id1, DataStore::CPU), 1, s1.Prod());
  basic::Reduction(in, out, closure);
  float* res = dstore.GetData(id2, DataStore::CPU);
  for (int i = 0; i < s2[0]; ++i) {
    for (int j = 0; j < s2[1]; ++j) {
      cout << res[i * s2[1] + j] << " ";
    }
    cout << endl;
  }
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(argc, argv);
  Test1();
  Test2();
  return 0;
}

