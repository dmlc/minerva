#include "minerva_system.h"

namespace minerva {

void MinervaSystem::Eval(NArray narr) {
  std::vector<uint64_t> id_to_eval = {narr.data_node_->node_id_};
  expand_engine_.Process(logical_dag_, id_to_eval);
}

}
