#include "data_store.h"
#include <iostream>

using namespace minerva;
using namespace std;

int main() {
  cout << DataStore::Instance().GenerateDataID() << endl;
  cout << DataStore::Instance().GenerateDataID() << endl;
  cout << DataStore::Instance().GenerateDataID() << endl;
  cout << DataStore::Instance().GenerateDataID() << endl;
  cout << DataStore::Instance().GetData(0, DataStore::CPU) << endl;
  return 0;
}
