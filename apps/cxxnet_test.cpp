#include <dmlc/io.h>
#include <dmlc/logging.h>
#include "minerva.h"
#include "io/data.h"
#include "utils/config.h"

using namespace std;
using namespace cxxnet;
using namespace dmlc;

void InitIter(IIterator<DataBatch>* itr,
					const std::vector< std::pair< std::string, std::string> > &defcfg) {
	for (size_t i = 0; i < defcfg.size(); ++ i) {
	  itr->SetParam(defcfg[i].first.c_str(), defcfg[i].second.c_str());
	}
	itr->Init();
}

IIterator<DataBatch>* CreateIterators(const std::vector< std::pair< std::string, std::string> >& cfg)
{
	IIterator<DataBatch>* data_itr;
	int flag = 0;
	std::string evname;
	std::vector< std::pair< std::string, std::string> > itcfg;
	std::vector< std::pair< std::string, std::string> > defcfg;
	for (size_t i = 0; i < cfg.size(); ++ i) {
	  const char *name = cfg[i].first.c_str();
	  const char *val  = cfg[i].second.c_str();

	  std::cout << flag << " " << name << " " << val << std::endl;

	  if (!strcmp(name, "data")) {
		flag = 1; continue;
	  }
	  if (!strcmp(name, "eval")) {
		flag = 2; continue;
	  }
	  if (!strcmp(name, "pred")) {
		flag = 3; continue;
	  }
	  if (!strcmp(name, "iter") && !strcmp(val, "end")) {
		if (flag == 1) {
		  data_itr = cxxnet::CreateIterator(itcfg);
		}
		flag = 0; itcfg.clear();
	  }
	  if (flag == 0) {
		defcfg.push_back(cfg[i]);
	  }else{
		itcfg.push_back(cfg[i]);
	  }
	}
	if (data_itr != NULL) {
	  InitIter(data_itr, defcfg);
	}
	return data_itr;
}


int main(int argc, char** argv) {
	std::vector< std::pair< std::string, std::string> > itcfg;
	dmlc::Stream *cfg = dmlc::Stream::Create(argv[1], "r");
	{
		dmlc::istream is(cfg);
		cxxnet::utils::ConfigStreamReader itr(is);
		itr.Init();
		while(itr.Next()) {
			std::cout << itr.name() << " " << itr.val() << std::endl;
			itcfg.push_back(std::make_pair(std::string(itr.name()), std::string(itr.val())));
		}
	}

	delete cfg;
	//Get the data and init
	IIterator<DataBatch>* data_itr = CreateIterators(itcfg);
	data_itr->BeforeFirst();
	int batch_dir = 0;
	while(data_itr->Next())
	{
		const DataBatch& batch = data_itr->Value();
		std::cout  << "Batch " << batch_dir ++ << " " << batch.batch_size << std::endl;
		mshadow::Tensor<mshadow::cpu, 2> label = batch.label;
		for(int i = 0; i < label.shape_.shape_[0]; i++)
			std::cout << label.dptr_[i] << " ";
		exit(0);
	}
}
