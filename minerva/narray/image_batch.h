#pragma once

namespace minerva {

class ImageBatch {
 public:
  NArray GetImage(int);


 private:
  int num_images_;
  int num_feature_maps_;
  int height_;
  int width_;
};

class Filter {
 public:

 private:
  int num_outputs_;
  int num_inputs_;
  int height_;
  int width_;
};

struct ConvInfo {
  int pad_height;
  int pad_width;
  int stride_vertical;
  int stride_horizontal;
};


}  // namespace minerva

