#include "basic.h"

#include <cmath>
#include <glog/logging.h>

using namespace std;

namespace minerva {
namespace basic {

void Arithmetic(DataList& inputs, DataList& outputs, ArithmeticClosure& closure) {
  CHECK_EQ(inputs.size(), 2) << "(arithmetic) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(arithmetic) #outputs is wrong!";
  float* left_data = inputs[0].GetCpuData();
  float* right_data = inputs[1].GetCpuData();
  float* res_data = outputs[0].GetCpuData();
  int length = outputs[0].Size().Prod();
  switch(closure.type) {
    case ADD:
      for (int i = 0; i < length; ++i) {
        res_data[i] = left_data[i] + right_data[i];
      }
      break;
    case SUB:
      for (int i = 0; i < length; ++i) {
        res_data[i] = left_data[i] - right_data[i];
      }
      break;
    case MULT:
      for (int i = 0; i < length; ++i) {
        res_data[i] = left_data[i] * right_data[i];
      }
      break;
    case DIV:
      for (int i = 0; i < length; ++i) {
        res_data[i] = left_data[i] / right_data[i];
      }
      break;
  }
}

void ArithmeticConst(DataList& inputs, DataList& outputs, ArithmeticConstClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(arithmetic const) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(arithmetic const) #outputs is wrong!";
  float val = closure.val;
  float* in_data = inputs[0].GetCpuData();
  float* res_data = outputs[0].GetCpuData();
  int length = outputs[0].Size().Prod();
  switch(closure.type) {
    case ADD:
      if(closure.side == 0) {// const on left
        for (int i = 0; i < length; ++i) {
          res_data[i] = in_data[i] + val;
        }
      } else {// const on right
        for (int i = 0; i < length; ++i) {
          res_data[i] = val + in_data[i];
        }
      }
      break;
    case SUB:
      if(closure.side == 0) {// const on left
        for (int i = 0; i < length; ++i) {
          res_data[i] = in_data[i] - val;
        }
      } else {// const on right
        for (int i = 0; i < length; ++i) {
          res_data[i] = val - in_data[i];
        }
      }
      break;
    case MULT:
      if(closure.side == 0) {// const on left
        for (int i = 0; i < length; ++i) {
          res_data[i] = in_data[i] * val;
        }
      } else {// const on right
        for (int i = 0; i < length; ++i) {
          res_data[i] = val * in_data[i];
        }
      }
      break;
    case DIV:
      if(closure.side == 0) {// const on left
        for (int i = 0; i < length; ++i) {
          res_data[i] = in_data[i] / val;
        }
      } else {// const on right
        for (int i = 0; i < length; ++i) {
          res_data[i] = val / in_data[i];
        }
      }
      break;
  }
}

void Elewise(DataList& inputs, DataList& outputs, ElewiseClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(elewise) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(elewise) #outputs is wrong!";
  float* in_data = inputs[0].GetCpuData();
  float* res_data = outputs[0].GetCpuData();
  int length = outputs[0].Size().Prod();
  switch(closure.type) {
    case EXP:
      for (int i = 0; i < length; ++i) {
        res_data[i] = exp(in_data[i]);
      }
      break;
    case LN:
      for (int i = 0; i < length; ++i) {
        res_data[i] = log(in_data[i]);
      }
      break;
    case SIGMOID:
      for (int i = 0; i < length; ++i) {
        res_data[i] = 1 / (1 + exp(-in_data[i]));
      }
      break;
    case NEGATIVE:
      for (int i = 0; i < length; ++i) {
        res_data[i] = -in_data[i];
      }
      break;
  }
}

void MatMult(DataList& inputs, DataList& outputs, MatMultClosure& closure) {
  CHECK_EQ(inputs.size(), 2) << "(matmult) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(matmult) #outputs is wrong!";
  float* left_data = inputs[0].GetCpuData();
  float* right_data = inputs[1].GetCpuData();
  float* res_data = outputs[0].GetCpuData();
  int m = outputs[0].Size()[0];
  int n = outputs[0].Size()[1];
  int o = inputs[0].Size()[1];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res_data[i * n + j] = 0;
      for (int k = 0; k < o; ++k) {
        res_data[i * n + j] += left_data[i * o + k] * right_data[k * n + j];
      }
    }
  }
}

void Transpose(DataList& inputs, DataList& outputs, TransposeClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(transpose) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(transpose) #outputs is wrong!";
  float* in_data = inputs[0].GetCpuData();
  float* res_data = outputs[0].GetCpuData();
  int m = outputs[0].Size()[0];
  int n = outputs[0].Size()[1];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res_data[i * n + j] = in_data[j * m + i];
    }
  }
}

void Reduction(DataList& inputs, DataList& outputs, ReductionClosure& closure) {
  CHECK_EQ(inputs.size(), 1) << "(reduction) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(reduction) #outputs is wrong!";
  float* in_data = inputs[0].GetCpuData();
  float* res_data = outputs[0].GetCpuData();
  for (auto i: closure.dims_to_reduce) {
  }
}


void Randn(DataShard& output, RandnClosure& closure) {
  int length = output.Size().Prod();
  float* data = output.GetCpuData();
  default_random_engine generator;
  normal_distribution<float> distribution(closure.mu, closure.var); // TODO only float for now
  for (int i = 0; i < length; ++i) {
    data[i] = distribution(generator);
  }
}

void Fill(DataShard& output, FillClosure& closure) {
  int length = output.Size().Prod();
  float* data = output.GetCpuData();
  for (int i = 0; i < length; ++i) {
    data[i] = closure.val;
  }
}

} // end of namespace basic
} // end of namespace minerva
