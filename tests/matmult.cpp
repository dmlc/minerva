#include <functional>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>

#include "procedures/dag_engine.h"
#include "dag/dag.h"
#include "chunk/chunk.h"

using namespace minerva;
using namespace std;

int main() {
  Chunk a = Chunk::Constant({100, 200}, 0.2);
  Chunk b = Chunk::Constant({200, 50}, 0.1);
  Chunk c = a * b;
  a.Print();
  c.Print();
  c.Print();
  return 0;
}
