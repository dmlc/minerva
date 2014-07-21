#include "physical_data.h"
#include "system/minerva_system.h"

namespace minerva {

PhysicalData::PhysicalData() {
}

PhysicalData::PhysicalData(const Scale& size): size(size) {
  data_id = MinervaSystem::Instance().data_store().GenerateDataID();
}

}

