#pragma once
#include <cstring>

namespace minerva {

class NArray;

class DBLoader {
 public:
  DBLoader(const std::string& dbname);
  void LoadNext(int stepsize);
  NArray GetData();
  NArray GetLabel();
};

}
