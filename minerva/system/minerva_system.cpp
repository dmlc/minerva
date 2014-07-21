#include "minerva_system.h"

using namespace std;

namespace minerva {

void MinervaSystem::Eval(NArray narr) {
  std::vector<uint64_t> id_to_eval = {narr.data_node_->node_id_};
  expand_engine_.Process(logical_dag_, id_to_eval);
  auto physical_nodes = expand_engine_.GetPhysicalNodes(narr.data_node_->node_id_);
  physical_engine_.Process(physical_dag_, physical_nodes);
}

}
