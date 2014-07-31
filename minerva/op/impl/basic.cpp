#include "basic.h"
#include "common/nvector.h"

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
  // TODO
  assert(false);
}


void Randn(DataList& output, RandnClosure& closure) {
  CHECK_EQ(output.size(), 1) << "wrong number of randn output";
  int length = output[0].Size().Prod();
  float* data = output[0].GetCpuData();
  default_random_engine generator;
  normal_distribution<float> distribution(closure.mu, closure.var); // TODO only float for now
  for (int i = 0; i < length; ++i) {
    data[i] = distribution(generator);
  }
}

void Fill(DataList& output, FillClosure& closure) {
  CHECK_EQ(output.size(), 1) << "wrong number of fill constant output";
  int length = output[0].Size().Prod();
  float* data = output[0].GetCpuData();
  for (int i = 0; i < length; ++i) {
    data[i] = closure.val;
  }
}

void Assemble(NVector<DataShard>& data_shards, float* dest, const Scale& dest_size) {
  Scale num_shards = data_shards.Size();
  size_t num_dims = num_shards.NumDims();
  NVector<Scale> shard_copysize = data_shards.Map<Scale>(
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
    Scale& copysize = shard_copysize[shard_index];
    Scale shard_copy_start = Scale::Origin(num_dims);
    ScaleRange localrange = ScaleRange::MakeRangeFromOrigin(ds.Size());
    //cout << "grange=" << globalrange << " lrange=" << localrange << endl;
    do {
      //cout << "off=" << ds.Offset() << " start=" << shard_copy_start << endl;
      size_t srcoff = localrange.Flatten(shard_copy_start);
      size_t dstoff = globalrange.Flatten(ds.Offset() + shard_copy_start);
      size_t len = copysize.Prod();
      //cout << "srcoff=" << srcoff << " dstoff=" << dstoff << " len=" << len << endl;
      // do copy
      memcpy(dest + dstoff, ds.GetCpuData() + srcoff, len * sizeof(float));
      ++copy_times;
      // incr copy_start
      shard_copy_start = shard_copy_start + copysize;
      for(size_t i = 0; i < num_dims; ++i)
        shard_copy_start[i] -= 1; // similar to "end = start + len - 1"
    } while(Scale::IncrOne(shard_copy_start, ds.Size()));
  } while(Scale::IncrOne(shard_index, num_shards));
  VLOG(1) << "copy times in assemble: " << copy_times;
}

void NCopy(float* src, const Scale& srcsize, const Scale& srcstart,
    float* dst, const Scale& dstsize, const Scale& dststart,
    const Scale& copysize) {
  size_t numdims = srcsize.NumDims();
  CHECK_EQ(srcstart.NumDims(), numdims) << "copy error: wrong #dims";
  CHECK_EQ(copysize.NumDims(), numdims) << "copy error: wrong #dims";
  CHECK_EQ(dstsize.NumDims(), numdims) << "copy error: wrong #dims";
  CHECK_EQ(dststart.NumDims(), numdims) << "copy error: wrong #dims";
  Scale srcend = srcstart + copysize;
  Scale dstend = dststart + copysize;
  CHECK_LE(dstend, dstsize) << "copy error: not enough dest space";
  Scale percopysize = Scale::Constant(numdims, 1);
  for(size_t i = 0; i < numdims; ++i) {
    percopysize[i] = copysize[i];
    if(!(srcstart[i] == 0 && srcend[i] == srcsize[i]
          && dststart[i] == 0 && dstend[i] == dstsize[i])) // remainings are non-contigous parts
      break;
  }
  Scale copystart = Scale::Origin(numdims);
  ScaleRange srcrange = ScaleRange::MakeRangeFromOrigin(srcsize);
  ScaleRange dstrange = ScaleRange::MakeRangeFromOrigin(dstsize);
  int copytimes = 0;
  int percopylen = percopysize.Prod();
  do {
    size_t srcoff = srcrange.Flatten(srcstart + copystart);
    size_t dstoff = dstrange.Flatten(dststart + copystart);
    // do memcopy
    memcpy(dst + dstoff, src + srcoff, percopylen * sizeof(float));
    ++copytimes;
    // incr copy_start
    copystart = copystart + percopysize - 1; // similar to "end = start + len - 1"
  } while(Scale::IncrOne(copystart, copysize));
  cout << "Copy times in NCopy:" << copytimes << endl;
}

} // end of namespace basic
} // end of namespace minerva
