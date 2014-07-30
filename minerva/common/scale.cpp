#include "scale.h"
#include "nvector.h"

namespace minerva {

const Scale Scale::kNullScale;
const ScaleRange ScaleRange::kNullRange;

int Scale::Prod() const {
  if(vec_.empty())
    return 0;
  else {
    int prod = 1;
    for(int i : vec_) prod *= i;
    return prod;
  }
}
std::string Scale::ToString() const {
  std::stringstream ss;
  ss << "[";
  for(size_t i = 0; i < vec_.size(); ++i)
    ss << vec_[i] << " ";
  ss << "]";
  return ss.str();
}

Scale operator + (const Scale& sc1, const Scale& sc2) {
	assert(sc1.NumDims() == sc2.NumDims());
	std::vector<int> vec;
	for(size_t i = 0; i < sc1.NumDims(); ++i)
		vec.push_back(sc1[i] + sc2[i]);
	return Scale(vec);
}
Scale operator - (const Scale& sc1, const Scale& sc2) {
	assert(sc1.NumDims() == sc2.NumDims());
	std::vector<int> vec;
	for(size_t i = 0; i < sc1.NumDims(); ++i)
		vec.push_back(sc1[i] - sc2[i]);
	return Scale(vec);
}
Scale operator * (const Scale& sc1, const Scale& sc2) {
	assert(sc1.NumDims() == sc2.NumDims());
	std::vector<int> vec;
	for(size_t i = 0; i < sc1.NumDims(); ++i)
		vec.push_back(sc1[i] * sc2[i]);
	return Scale(vec);
}
Scale operator / (const Scale& sc1, const Scale& sc2) {
	assert(sc1.NumDims() == sc2.NumDims());
	std::vector<int> vec;
	for(size_t i = 0; i < sc1.NumDims(); ++i)
		vec.push_back(sc1[i] / sc2[i]);
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
  assert(numparts.NumDims() == NumDims());
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

} // end of namespace minerva
