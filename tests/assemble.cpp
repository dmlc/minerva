#include <minerva.h>
#include <iostream>
#include <op/impl/basic.h>

using namespace minerva;
using namespace std;

void Fill(float* arr, float val, size_t len) {
  for(size_t i = 0; i < len; ++i) arr[i] = val;
}

void Test1() {
  // prepare
  DataStore& dstore = MinervaSystem::Instance().data_store();
  Scale s1 = {2, 3}, s2 = {2, 3}, s3 = {2, 3}, s4 = {2, 3};
  Scale o1 = {0, 0}, o2 = {0, 3}, o3 = {2, 0}, o4 = {2, 3};
  Scale oi1 = {0, 0}, oi2 = {0, 1}, oi3 = {1, 0}, oi4 = {1, 1};
  Scale srst = {4, 6};
  uint64_t id1 = dstore.GenerateDataID();
  uint64_t id2 = dstore.GenerateDataID();
  uint64_t id3 = dstore.GenerateDataID();
  uint64_t id4 = dstore.GenerateDataID();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  dstore.CreateData(id3, DataStore::CPU, s3.Prod());
  dstore.CreateData(id4, DataStore::CPU, s4.Prod());
  Fill(dstore.GetData(id1, DataStore::CPU), 1, s1.Prod());
  Fill(dstore.GetData(id2, DataStore::CPU), 2, s2.Prod());
  Fill(dstore.GetData(id3, DataStore::CPU), 3, s3.Prod());
  Fill(dstore.GetData(id4, DataStore::CPU), 4, s4.Prod());
  NVector<PhysicalData> dvec({2, 2});
  dvec[oi1] = {s1, o1, oi1, id1, NULL};
  dvec[oi2] = {s2, o2, oi2, id2, NULL};
  dvec[oi3] = {s3, o3, oi3, id3, NULL};
  dvec[oi4] = {s4, o4, oi4, id4, NULL};
  float * rst = new float[srst.Prod()];
  NVector<DataShard> ds = dvec.Map<DataShard>([] (const PhysicalData& pd) { return DataShard(pd); });
  // assemble
  basic::Assemble(ds, rst, srst);

  // print
  for(int i = 0; i < srst[0]; ++i) {
    for(int j = 0; j < srst[1]; ++j) {
      cout << rst[i + srst[0] * j] << " ";
    }
    cout << endl;
  }
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(argc, argv);
  Test1();
  return 0;
}
