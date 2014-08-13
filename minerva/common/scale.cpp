#include "scale.h"
#include "nvector.h"
#include <glog/logging.h>
#include <algorithm>

using namespace std;

namespace minerva {

////////////////////////////////////////////////////////
// method definitions for class: Scale
////////////////////////////////////////////////////////
const Scale Scale::kNullScale;

string Scale::ToString() const {
  stringstream ss;
  ss << "[";
  for(size_t i = 0; i < vec_.size(); ++i)
    ss << vec_[i] << " ";
  ss << "]";
  return ss.str();
}

Scale operator + (const Scale& sc1, const Scale& sc2) {
	CHECK_EQ(sc1.NumDims(), sc2.NumDims()) << "dimension mismatch";
	vector<int> vec;
	for(size_t i = 0; i < sc1.NumDims(); ++i)
		vec.push_back(sc1[i] + sc2[i]);
	return Scale(vec);
}
Scale operator - (const Scale& sc1, const Scale& sc2) {
	CHECK_EQ(sc1.NumDims(), sc2.NumDims()) << "dimension mismatch";
	vector<int> vec;
	for(size_t i = 0; i < sc1.NumDims(); ++i)
		vec.push_back(sc1[i] - sc2[i]);
	return Scale(vec);
}
Scale operator * (const Scale& sc1, const Scale& sc2) {
	CHECK_EQ(sc1.NumDims(), sc2.NumDims()) << "dimension mismatch";
	vector<int> vec;
	for(size_t i = 0; i < sc1.NumDims(); ++i)
		vec.push_back(sc1[i] * sc2[i]);
	return Scale(vec);
}
Scale operator / (const Scale& sc1, const Scale& sc2) {
	CHECK_EQ(sc1.NumDims(), sc2.NumDims()) << "dimension mismatch";
	vector<int> vec;
	for(size_t i = 0; i < sc1.NumDims(); ++i)
		vec.push_back(sc1[i] / sc2[i]);
	return Scale(vec);
}
Scale operator + (const Scale& sc1, int val) {
  vector<int> vec;
  for_each(sc1.vec_.begin(), sc1.vec_.end(), [&] (int x) { vec.push_back(x + val); });
  return Scale(vec);
}

Scale operator - (const Scale& sc1, int val) {
  vector<int> vec;
  for_each(sc1.vec_.begin(), sc1.vec_.end(), [&] (int x) { vec.push_back(x - val); });
  return Scale(vec);
}

Scale operator * (const Scale& sc1, int val) {
  vector<int> vec;
  for_each(sc1.vec_.begin(), sc1.vec_.end(), [&] (int x) { vec.push_back(x * val); });
  return Scale(vec);
}

Scale operator / (const Scale& sc1, int val) {
  vector<int> vec;
  for_each(sc1.vec_.begin(), sc1.vec_.end(), [&] (int x) { vec.push_back(x / val); });
  return Scale(vec);
}

bool Scale::IncrOne(const Scale& max) {
  for (size_t i = 0; i < NumDims(); ++i) {
    if (vec_[i] + 1 < max[i]) {
      ++vec_[i];
      return true;
    } else {
      vec_[i] = 0;
    }
  }
  return false;
}

bool Scale::IncrOne(Scale& pos, const Scale& max) {
  return pos.IncrOne(max);
}

bool Scale::IncrWithDimensionsFixed(const Scale& max, const Scale& fix) {
  size_t num = NumDims();
  for (size_t i = 0; i < num; ++i) {
    if (fix.Contains(i)) {
      continue;
    }
    if (vec_[i] + 1 < max[i]) {
      ++vec_[i];
      return true;
    } else {
      vec_[i] = 0;
    }
  }
  return false;
}

bool Scale::IncrDimensions(const Scale& max, const Scale& fix) {
  size_t num = NumDims();
  for (size_t i = 0; i < num; ++i) {
    if (!fix.Contains(i)) {
      continue;
    }
    if (vec_[i] + 1 < max[i]) {
      ++vec_[i];
      return true;
    } else {
      vec_[i] = 0;
    }
  }
  return false;
}

NVector<Scale> Scale::EquallySplit(const Scale& numparts) const {
  CHECK_EQ(numparts.NumDims(), NumDims()) << "partition dimension is wrong";
  NVector<Scale> rst(numparts);
  Scale partsize = *this / numparts;
  Scale pos = Scale::Origin(NumDims());
  do {
    Scale size = partsize;
    for(size_t i = 0; i < NumDims(); ++i) {
      if(pos[i] == numparts[i] - 1) { // last partition
        size[i] = (*this)[i] - pos[i] * size[i];
      }
    }
    rst[pos] = size;
  } while(pos.IncrOne(numparts));
  return rst;
}

Scale Scale::Merge(const NVector<Scale>& partsizes) {
  size_t numdims = partsizes.Size().NumDims();
  Scale rst = Scale::Origin(numdims);
  for(size_t dim = 0; dim < numdims; ++dim) {
    Scale pos = Scale::Origin(numdims);
    for(int i = 0; i < partsizes.Size()[dim]; ++i) {
      rst[dim] += partsizes[pos][dim];
      ++pos[dim];
    }
  }
  return rst;
}
  
Scale Scale::Concat(int val) const {
  Scale ret(*this);
  ret.vec_.push_back(val);
  return ret;
}
  
NVector<Scale> Scale::ToNVector() const {
  Scale size = Scale::Constant(NumDims(), 1);
  return NVector<Scale>({*this}, size);
}

////////////////////////////////////////////////////////
// method definitions for class: ScaleRange
////////////////////////////////////////////////////////
const ScaleRange ScaleRange::kNullRange;

bool ScaleRange::ValidRange(const Scale& st, const Scale& ed) {
  bool valid = st.NumDims() == ed.NumDims();
  if(valid) {
    for(size_t i = 0; valid && i < st.NumDims(); ++i) {
      valid &= (st[i] <= ed[i]);
    }
  }
  return valid;
}
ScaleRange ScaleRange::Intersect(const ScaleRange& r1, const ScaleRange& r2) {
  if(r1.NumDims() != r2.NumDims()) return kNullRange;
  vector<int> new_st, new_ed;
  for(size_t i = 0; i < r1.NumDims(); ++i) {
    new_st.push_back(max(r1.start_[i], r2.start_[i]));
    new_ed.push_back(min(r1.end_[i], r2.end_[i]));
  }
  return MakeRange(Scale(new_st), Scale(new_ed));
}

size_t ScaleRange::Area() const {
  size_t area = 1;
  bool all_zero = true;
  for(size_t i = 0; i < start_.NumDims(); ++i) {
    if(end_[i] != start_[i]) {
      area *= end_[i] - start_[i];
      all_zero = false;
    }
  }
  return all_zero ? 0 : area;
}

size_t ScaleRange::Flatten(const Scale& sc) const {
  CHECK(IsInRange(sc));
  size_t stride = 1;
  size_t ret = 0;
  for(size_t i = 0; i < sc.NumDims(); ++i) {
    ret += (sc[i] - start_[i]) * stride;
    stride *= end_[i] - start_[i];
  }
  return ret;
}

} // end of namespace minerva
