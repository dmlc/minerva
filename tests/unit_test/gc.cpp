#include <minerva.h>

using namespace minerva;
using namespace std;

void Test() {
  NArray narr = NArray::Constant({10, 8}, 0.0, {2, 1});
  for(int i = 0; i < 10; ++i) {
    cout << "####iter " << i << endl;
    narr += 1;
    narr.Eval();
    //cout << MinervaSystem::Instance().logical_dag().PrintDag<ExternRCPrinter>() << endl;
    //cout << MinervaSystem::Instance().physical_dag().PrintDag<ExternRCPrinter>() << endl;
  }
  float* val = narr.Get();
  for(int i = 0; i < 5; ++i)
    cout << val[i] << " ";
  cout << endl;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(argc, argv);
  Test();
  return 0;
}
