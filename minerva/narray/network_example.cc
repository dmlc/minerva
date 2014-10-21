ImageBatch l1 = NArray::LoadFromFile(...);

ConvInfo conv2{0, 0, 4, 4};  // No padding, stride of 4
Filter f2 = NArray::Randn({11, 11, 3, 96}, 0.0, 1.0);  // Kernel size of 11, 3 input channels, 96 output channels
ImageBatch l2 = Convolution::ConvForward(l1, f2, NArray::Zeros({3, 96}), conv2);

ImageBatch l3 = Convolution::ActivationForward(l2, ActivationAlgorithm::kRelu);

PoolingInfo pool4{PoolingInfo::kMax, 3, 3, 2, 2};  // Max pooling, kernel size = 3, stride = 2
ImageBatch l4 = Convolution::PoolingForward(l3, pool4);

// ... A lot more

ImageBatch l4_diff = ...;
ImageBatch l3_diff = Convolution::PoolingBackward(l4_diff, l4, l3, pool4);
ImageBatch l2_diff = Convolution::ActivationBackward(l3_diff, l3, l2, ActivationAlgorithm::kRelu);
Filter f1_diff = Convolution::ConvBackwardFilter(l2_diff, l1, conv2);
ImageBatch l1_diff = Convolution::ConvBackwardData(l2_diff, f2, conv2);
// Add bias as normal NArray operation
