#include "basic.h"
#include "common/nvector.h"

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
  auto in_max = inputs[0].Size();
  auto in_range = ScaleRange::MakeRangeFromOrigin(in_max);
  auto res_max = outputs[0].Size();
  auto res_range = ScaleRange::MakeRangeFromOrigin(res_max);
  auto accumulator = Scale::Origin(max.NumDims());
  do {
    auto cur = accumulator;
    float tmp = in_data[in_range.Flatten(cur)];
    while (cur.IncrDimensions(in_max, closure.dims_to_reduce)) {
      float tmp2 = in_data[in_range.Flatten(cur)];
      switch (closure.type) {
        case SUM:
          tmp += tmp2;
          break;
        case MAX:
          if (tmp < tmp2) {
            tmp = tmp2;
          }
          break;
      }
    }
    res_data[res_range.Flatten(accumulator)] = tmp;
  } while (accumulator.IncrWithDimensionsFixed(res_max, closure.dims_to_reduce));
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

void Assemble(NVector<DataShard>& data_shards, float* dest, const Scale& dest_size) {
  Scale num_shards = data_shards.Size();
  size_t num_dims = num_shards.NumDims();
  NVector<Scale> shard_copy_size = data_shards.Map<Scale>(
      [&] (const DataShard& ds) {
        Scale ret = Scale::Constant(num_dims, 1);
        for(size_t i = 0; i < num_dims; ++i) {
          ret[i] = ds.Size()[i];
          if(num_shards[i] != 1)
            break;
        }
        return ret;
      }
    );
  // Copy each shard to dest
  Scale shard_index = Scale::Origin(num_dims);
  ScaleRange globalrange = ScaleRange::MakeRangeFromOrigin(dest_size);
  int copy_times = 0;
  do {
    DataShard& ds = data_shards[shard_index];
    Scale& copy_size = shard_copy_size[shard_index];
    Scale shard_copy_start = Scale::Origin(num_dims);
    ScaleRange localrange = ScaleRange::MakeRangeFromOrigin(ds.Size());
    //cout << "grange=" << globalrange << " lrange=" << localrange << endl;
    do {
      //cout << "off=" << ds.Offset() << " start=" << shard_copy_start << endl;
      size_t srcoff = localrange.Flatten(shard_copy_start);
      size_t dstoff = globalrange.Flatten(ds.Offset() + shard_copy_start);
      size_t len = copy_size.Prod();
      //cout << "srcoff=" << srcoff << " dstoff=" << dstoff << " len=" << len << endl;
      // do copy
      memcpy(dest + dstoff, ds.GetCpuData() + srcoff, len * sizeof(float));
      ++copy_times;
      // incr copy_start
      shard_copy_start = shard_copy_start + copy_size;
      for(size_t i = 0; i < num_dims; ++i)
        shard_copy_start[i] -= 1; // similar to "end = start + len - 1"
    } while(shard_copy_start.IncrOne(ds.Size()));
  } while(shard_index.IncrOne(num_shards));
  cout << "copy times: " << copy_times << endl;
}

} // end of namespace basic
} // end of namespace minerva
