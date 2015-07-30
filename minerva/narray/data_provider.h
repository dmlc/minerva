#pragma once
#include "narray/image_batch.h"
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include "minerva.h"
#include "utils/config.h"

using namespace std;
using namespace cxxnet;
using namespace dmlc;

namespace cxxnet {
class DataBatch;
template<typename T> class IIterator;
}

namespace minerva {

class DataProvider {
 public:
   cxxnet::IIterator<cxxnet::DataBatch>* data_itr;
	 DataProvider(std::string dataconfig);
	 ~DataProvider();
	 vector<NArray> GetNextValue();
 private:
	 void CreateDataIterator();
	 void Init();
	 std::vector< std::pair< std::string, std::string> > cfg;
	 Scale data_scale, label_scale;
	 int batch_size_; 
};


}  // namespace minerva