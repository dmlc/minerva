#pragma once
#include <vector>

namespace minerva {

/* Order `TaskType` from higher to lower priority. 
 */
enum class TaskType {
  kToRun,
  kToComplete,
  kToDelete,
  kEnd  // Sentry
};

}  // namespace minerva

