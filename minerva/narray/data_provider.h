#pragma once
#include <vector>
#include <utility>
#include <cstring>
#include <narray/narray.h>

namespace cxxnet {
class DataBatch;
template<typename T> class IIterator;
}

namespace minerva {

class DataProvider {
 public:
   cxxnet::IIterator<cxxnet::DataBatch>* data_itr;
	 DataProvider(const std::string &dataconfig);
	 ~DataProvider();
   std::vector<NArray> GetNextValue();
 private:
	 void CreateDataIterator();
	 void Init();
	 std::vector< std::pair< std::string, std::string> > cfg;
	 Scale data_scale, label_scale;
	 int batch_size_; 
};


}  // namespace minerva
