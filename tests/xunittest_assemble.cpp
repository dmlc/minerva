#include <minerva.h>
#include <iostream>
#include <op/impl/basic.h>

using namespace minerva;
using namespace std;

PhysicalData MakeData(Scale s, Scale o, Scale oi, uint64_t id) {
  PhysicalData pdata;
  pdata.size = s;
  pdata.offset = o;
  pdata.offset_index = oi;
  pdata.data_id = id;
  return pdata;
}

void Fill(float* arr, float val, size_t len) {
  for(size_t i = 0; i < len; ++i) arr[i] = val;
}

void Test1() {
  cout << "Test 2D assemble" << endl;
  // prepare
  // make input
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
  vector<PhysicalData> dvec;
  dvec.push_back(MakeData(s1, o1, oi1, id1));
  dvec.push_back(MakeData(s2, o2, oi2, id2));
  dvec.push_back(MakeData(s3, o3, oi3, id3));
  dvec.push_back(MakeData(s4, o4, oi4, id4));
  vector<DataShard> inds;
  for_each(dvec.begin(), dvec.end(), [&] (PhysicalData& pd) { inds.push_back(DataShard(pd)); });
  // make output
  uint64_t rstid = dstore.GenerateDataID();
  dstore.CreateData(rstid, DataStore::CPU, srst.Prod());
  PhysicalData rstpd; rstpd.size = srst; rstpd.data_id = rstid;
  vector<DataShard> outds{DataShard(rstpd)};
  // assemble
  AssembleClosure ac;
  basic::Assemble(inds, outds, ac);

  // print
  float* rst = outds[0].GetCpuData();
  for(int i = 0; i < srst[0]; ++i) {
    for(int j = 0; j < srst[1]; ++j) {
      cout << rst[i + srst[0] * j] << " ";
    }
    cout << endl;
  }
}

void Test2() {
  cout << "Test 3D assemble (split in 1st dim)" << endl;
  DataStore& dstore = MinervaSystem::Instance().data_store();
  // make input
  Scale s1 = {2, 6, 8}, s2 = {2, 6, 8};
  Scale o1 = {0, 0, 0}, o2 = {2, 0, 0};
  Scale oi1 = {0, 0, 0}, oi2 = {1, 0, 0};
  Scale srst = {4, 6, 8};
  uint64_t id1 = dstore.GenerateDataID();
  uint64_t id2 = dstore.GenerateDataID();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  Fill(dstore.GetData(id1, DataStore::CPU), 1, s1.Prod());
  Fill(dstore.GetData(id2, DataStore::CPU), 2, s2.Prod());
  vector<PhysicalData> dvec;
  dvec.push_back(MakeData(s1, o1, oi1, id1));
  dvec.push_back(MakeData(s2, o2, oi2, id2));
  vector<DataShard> inds;
  for_each(dvec.begin(), dvec.end(), [&] (PhysicalData& pd) { inds.push_back(DataShard(pd)); });
  // make output
  uint64_t rstid = dstore.GenerateDataID();
  dstore.CreateData(rstid, DataStore::CPU, srst.Prod());
  PhysicalData rstpd; rstpd.size = srst; rstpd.data_id = rstid;
  vector<DataShard> outds{DataShard(rstpd)};
  // assemble
  AssembleClosure ac;
  basic::Assemble(inds, outds, ac);

  // print
  float* rst = outds[0].GetCpuData();
  for(int i = 0; i < 4; ++i) {
    for(int j = 0; j < 6; ++j) {
      cout << rst[i + j * 4] << " ";
    }
    cout << endl;
  }
}

void Test3() {
  cout << "Test 3D assemble (split in 2nd dim)" << endl;
  DataStore& dstore = MinervaSystem::Instance().data_store();
  // make input
  Scale s1 = {4, 3, 8}, s2 = {4, 3, 8};
  Scale o1 = {0, 0, 0}, o2 = {0, 3, 0};
  Scale oi1 = {0, 0, 0}, oi2 = {0, 1, 0};
  Scale srst = {4, 6, 8};
  Scale numparts = {1, 2 ,1};
  uint64_t id1 = dstore.GenerateDataID();
  uint64_t id2 = dstore.GenerateDataID();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  Fill(dstore.GetData(id1, DataStore::CPU), 1, s1.Prod());
  Fill(dstore.GetData(id2, DataStore::CPU), 2, s2.Prod());
  vector<PhysicalData> dvec;
  dvec.push_back(MakeData(s1, o1, oi1, id1));
  dvec.push_back(MakeData(s2, o2, oi2, id2));
  vector<DataShard> inds;
  for_each(dvec.begin(), dvec.end(), [&] (PhysicalData& pd) { inds.push_back(DataShard(pd)); });
  // make output
  uint64_t rstid = dstore.GenerateDataID();
  dstore.CreateData(rstid, DataStore::CPU, srst.Prod());
  PhysicalData rstpd; rstpd.size = srst; rstpd.data_id = rstid;
  vector<DataShard> outds{DataShard(rstpd)};
  // assemble
  AssembleClosure ac;
  basic::Assemble(inds, outds, ac);

  // print
  float* rst = outds[0].GetCpuData();
  for(int i = 0; i < 4; ++i) {
    for(int j = 0; j < 6; ++j) {
      cout << rst[i + j * 4] << " ";
    }
    cout << endl;
  }
}

void Test4() {
  cout << "Test 3D assemble (split in 3rd dim)" << endl;
  DataStore& dstore = MinervaSystem::Instance().data_store();
  // make input
  Scale s1 = {4, 6, 4}, s2 = {4, 6, 4};
  Scale o1 = {0, 0, 0}, o2 = {0, 0, 4};
  Scale oi1 = {0, 0, 0}, oi2 = {0, 0, 1};
  Scale srst = {4, 6, 8};
  Scale numparts = {1, 1 ,2};
  uint64_t id1 = dstore.GenerateDataID();
  uint64_t id2 = dstore.GenerateDataID();
  dstore.CreateData(id1, DataStore::CPU, s1.Prod());
  dstore.CreateData(id2, DataStore::CPU, s2.Prod());
  Fill(dstore.GetData(id1, DataStore::CPU), 1, s1.Prod());
  Fill(dstore.GetData(id2, DataStore::CPU), 2, s2.Prod());
  vector<PhysicalData> dvec;
  dvec.push_back(MakeData(s1, o1, oi1, id1));
  dvec.push_back(MakeData(s2, o2, oi2, id2));
  vector<DataShard> inds;
  for_each(dvec.begin(), dvec.end(), [&] (PhysicalData& pd) { inds.push_back(DataShard(pd)); });
  // make output
  uint64_t rstid = dstore.GenerateDataID();
  dstore.CreateData(rstid, DataStore::CPU, srst.Prod());
  PhysicalData rstpd; rstpd.size = srst; rstpd.data_id = rstid;
  vector<DataShard> outds{DataShard(rstpd)};
  // assemble
  AssembleClosure ac;
  basic::Assemble(inds, outds, ac);

  // print
  float* rst = outds[0].GetCpuData();
  cout << "front: " << endl;
  for(int i = 0; i < 4; ++i) {
    for(int j = 0; j < 6; ++j) {
      cout << rst[i + j * 4] << " ";
    }
    cout << endl;
  }
  cout << "back: " << endl;
  for(int i = 0; i < 4; ++i) {
    for(int j = 0; j < 6; ++j) {
      cout << rst[95 + i + j * 4] << " ";
    }
    cout << endl;
  }
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(&argc, &argv);
  Test1();
  Test2();
  Test3();
  Test4();
  return 0;
}
