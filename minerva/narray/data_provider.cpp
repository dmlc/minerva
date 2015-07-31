#include <dmlc/io.h>
#include <dmlc/logging.h>

#include "narray/data_provider.h"
#include "op/physical_op.h"
#include "io/data.h"
#include "utils/config.h"

using namespace std;
using namespace cxxnet;
using namespace dmlc;

namespace minerva {


void DataProvider::Init() {
	int flag = 0;
	std::string evname;
	std::vector< std::pair< std::string, std::string> > itcfg;
	std::vector< std::pair< std::string, std::string> > defcfg;
	for (size_t i = 0; i < cfg.size(); ++ i) {
	  const char *name = cfg[i].first.c_str();
	  const char *val  = cfg[i].second.c_str();
	  //std::cout << flag << " " << name << " " << val << std::endl;

	  if (!strcmp(name, "data")) {
		  flag = 1;
      continue;
	  }
	  if (!strcmp(name, "eval")) {
		  flag = 2;
      continue;
	  }
	  if (!strcmp(name, "pred")) {
		  flag = 3;
      continue;
	  }
	  if (!strcmp(name, "iter") && !strcmp(val, "end")) {
      if (flag == 1) {
        data_itr = cxxnet::CreateIterator(itcfg);
      }
		  flag = 0; itcfg.clear();
	  }
	  if (flag == 0) {
		  defcfg.push_back(cfg[i]);
	  } else{
		  itcfg.push_back(cfg[i]);
	  }
	}
	if (data_itr != NULL) {
    data_itr->SetParam("buffer_size", "5");
    for (size_t i = 0; i < defcfg.size(); ++ i) {
		  if (!strcmp(defcfg[i].first.c_str(), "batch_size")) {
			  batch_size_ = atoi(defcfg[i].second.c_str()) / num_gpu_;	
        std::ostringstream oss;
        oss << batch_size_;
		    data_itr->SetParam("batch_size", oss.str().c_str());
      } else {
		    data_itr->SetParam(defcfg[i].first.c_str(), defcfg[i].second.c_str());
      }
		  if (!strcmp(defcfg[i].first.c_str(), "input_shape")) {
			  sscanf(defcfg[i].second.c_str(), "%u,%u,%u", &channel, &width, &height);
      }
		  if (!strcmp(defcfg[i].first.c_str(), "label_dim")) {
			  sscanf(defcfg[i].second.c_str(), "%u", &label_dim);
      }
	  }
    std::vector<int> label_scale_vec;
    label_scale_vec.push_back(label_dim);
    label_scale_vec.push_back(batch_size_);
    label_scale = Scale(label_scale_vec);
	  std::vector<int> data_scale_vec;
    data_scale_vec.push_back(height);
    data_scale_vec.push_back(width);
    data_scale_vec.push_back(channel);
    data_scale_vec.push_back(batch_size_);
    data_scale = Scale(data_scale_vec);
    data_itr->Init();
	}
	data_itr->BeforeFirst();
}

DataProvider::DataProvider(const std::string &dataconfig, int num_gpu): num_gpu_(num_gpu) {
	dmlc::Stream *pcfg = dmlc::Stream::Create(dataconfig.c_str(), "r");
	{
		dmlc::istream is(pcfg);
		cxxnet::utils::ConfigStreamReader itr(is);
		itr.Init();
		while(itr.Next()) {
			std::cout << itr.name() << " " << itr.val() << std::endl;
			cfg.push_back(std::make_pair(std::string(itr.name()), std::string(itr.val())));
		}
	}
	delete pcfg;
	//Get the data and init
	Init();
}

DataProvider::~DataProvider() {}

vector<NArray> DataProvider::GetNextValue()
{
	CHECK_EQ(label_scale.NumDims(), 2) << "label scale error";
	CHECK_EQ(data_scale.NumDims(), 4) << "data scale error";
	DataProviderOp* op = new DataProviderOp();
	op->closure = {data_itr};
	vector<Scale> result_sizes;
	result_sizes.push_back(data_scale);
	result_sizes.push_back(label_scale);
	return NArray::Compute({}, result_sizes, op);
}

}  // namespace minerva
