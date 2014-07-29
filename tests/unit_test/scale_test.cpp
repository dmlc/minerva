#include <minerva.h>
#include <iostream>

using namespace std;
using namespace minerva;

void Test1() {
  cout << "Test flatten" << endl;
  Scale st = {0, 0}, ed = {4, 5};
  ScaleRange range = ScaleRange::MakeRange(st, ed);
  cout << range.Flatten({0, 0}) << " expected: 0" << endl;
  cout << range.Flatten({3, 4}) << " expected: 19" << endl;
  cout << range.Flatten({0, 4}) << " expected: 16" << endl;
  cout << range.Flatten({3, 0}) << " expected: 3" << endl;
  cout << range.Flatten({1, 2}) << " expected: 9" << endl;
  cout << range.Flatten({3, 2}) << " expected: 11" << endl;
}

void Test2() {
  cout << "Test flatten with non-origin start" << endl;
  Scale st = {1, 1}, ed = {5, 6};
  ScaleRange range = ScaleRange::MakeRange(st, ed);
  cout << range.Flatten({1, 1}) << " expected: 0" << endl;
  cout << range.Flatten({4, 5}) << " expected: 19" << endl;
  cout << range.Flatten({1, 5}) << " expected: 16" << endl;
  cout << range.Flatten({4, 1}) << " expected: 3" << endl;
  cout << range.Flatten({2, 3}) << " expected: 9" << endl;
  cout << range.Flatten({4, 3}) << " expected: 11" << endl;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(argc, argv);
  Test1();
  Test2();
  return 0;
}
