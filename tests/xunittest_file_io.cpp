#include <minerva.h>

using namespace minerva;
using namespace std;

void GenRandmat() {
  NArray na = NArray::Randn({4, 6}, 0.0, 1.0, {2, 2});
  FileFormat format;
  format.binary = false;
  na.ToFile("randmat.txt", format);
  format.binary = true;
  na.ToFile("randmat.dat", format);
}

void LoadAndOutput() {
  SimpleFileLoader loader;
  NArray na = NArray::LoadFromFile({4, 6}, "randmat.dat", &loader, {2, 2});

  FileFormat format;
  format.binary = false;
  na.ToFile("randmat1.txt", format);

  cout << "physical dag: " << endl;
  cout << MinervaSystem::Instance().physical_dag().PrintDag() << endl;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(&argc, &argv);
  //GenRandmat();
  LoadAndOutput();
  return 0;
}
