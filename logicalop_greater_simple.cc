/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include "third_party/tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "third_party/tensorflow/core/platform/logging.h"
#include "third_party/tensorflow/core/platform/stream_executor.h"
#include "third_party/tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/tensorrt/NvInfer.h"

using namespace nvinfer1;

namespace tensorflow {
namespace {

//#define RESULT_FLOAT

class Logger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    switch (severity) {
      case Severity::kINFO:
        LOG(INFO) << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << msg;
        break;
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LOG(ERROR) << msg;
        break;
      default:
        break;
    }
  }
};

const char* kInput1Tensor = "input1";
const char* kInput2Tensor = "input2";
const char* kOutputTensor = "output";

// Creates a network to compute y=x>x.
nvinfer1::IHostMemory* CreateNetwork() {
  Logger logger;
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

  nvinfer1::INetworkDefinition* network = builder->createNetwork();
  // Add the input.
  auto input1 = network->addInput(kInput1Tensor, nvinfer1::DataType::kFLOAT,
                                 nvinfer1::DimsCHW{1, 1, 1});
  EXPECT_NE(input1, nullptr);
  auto input2 = network->addInput(kInput2Tensor, nvinfer1::DataType::kFLOAT,
                                 nvinfer1::DimsCHW{1, 1, 1});
  EXPECT_NE(input2, nullptr);
  // Add the hidden layer.
  LOG(INFO) << "start layer";
  auto layer =
      network->addElementWise(*input1, *input2, ElementWiseOperation::kGREATER);
  EXPECT_NE(layer, nullptr);
  layer->setOutputType(0, nvinfer1::DataType::kBOOL); // This line doesn't seem to be useful
  LOG(INFO) << "finish layer";
  // Mark the output.
  auto output = layer->getOutput(0);
  output->setName(kOutputTensor);
  network->markOutput(*output);
  // Build the engine
  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(1 << 10);
  LOG(INFO) << "start engine";
  auto engine = builder->buildCudaEngine(*network);
  LOG(INFO) << "finish engine";
  EXPECT_NE(engine, nullptr);
  // Serialize the engine to create a model, then close everything.
  nvinfer1::IHostMemory* model = engine->serialize();
  network->destroy();
  engine->destroy();
  builder->destroy();
  return model;
}


TEST(TensorrtTest, LogicalGreater) {
  LOG(WARNING) << "TensorRT version " << NV_TENSORRT_MAJOR << "."
               << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH;

  // Handle the case where the test is run on machine with no gpu available.
  if (CHECK_NOTNULL(GPUMachineManager())->VisibleDeviceCount() <= 0) {
    LOG(WARNING) << "No gpu device available, probably not being run on a gpu "
                    "machine. Skipping...";
    return;
  }

  // Create the network model.
  nvinfer1::IHostMemory* model = CreateNetwork();
  (void)model;
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
