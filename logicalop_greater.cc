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

// Executes the network.
void Execute(nvinfer1::IExecutionContext* context, const float* input,
#ifdef RESULT_FLOAT
             float* output) {
#else
             bool* output) {
#endif
  const nvinfer1::ICudaEngine& engine = context->getEngine();

  // We have two bindings: input and output.
  ASSERT_EQ(engine.getNbBindings(), 2);
  const int input1_index = engine.getBindingIndex(kInput1Tensor);
  const int input2_index = engine.getBindingIndex(kInput2Tensor);
  const int output_index = engine.getBindingIndex(kOutputTensor);

  // Create GPU buffers and a stream
  void* buffers[3];
  ASSERT_EQ(0, cudaMalloc(&buffers[input1_index], sizeof(float)));
  ASSERT_EQ(0, cudaMalloc(&buffers[input2_index], sizeof(float)));
  LOG(INFO) << "start cudaMalloc result";
#ifdef RESULT_FLOAT
  ASSERT_EQ(0, cudaMalloc(&buffers[output_index], sizeof(float)));
#else
  ASSERT_EQ(0, cudaMalloc(&buffers[output_index], sizeof(bool)));
#endif
  LOG(INFO) << "finish cudaMalloc result";
  cudaStream_t stream;
  ASSERT_EQ(0, cudaStreamCreate(&stream));

  // Copy the input to the GPU, execute the network, and copy the output back.
  //
  // Note that since the host buffer was not created as pinned memory, these
  // async copies are turned into sync copies. So the following synchronization
  // could be removed.
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input1_index], input, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input2_index], input, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  context->enqueue(1, buffers, stream, nullptr);
  ASSERT_EQ(0, cudaMemcpyAsync(output, buffers[output_index], sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release the stream and the buffers
  cudaStreamDestroy(stream);
  ASSERT_EQ(0, cudaFree(buffers[input1_index]));
  ASSERT_EQ(0, cudaFree(buffers[input2_index]));
  ASSERT_EQ(0, cudaFree(buffers[output_index]));
}

// TODO(bixia): use typed test

TEST(TensorrtTest, WorkfloatFP32) {
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
  // Use the model to create an engine and then an execution context.
  Logger logger;
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
  nvinfer1::ICudaEngine* engine =
      runtime->deserializeCudaEngine(model->data(), model->size(), nullptr);
  model->destroy();
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();

  // Execute the network.
  float input = 12;
#ifdef RESULT_FLOAT
  float output;
#else
  bool output;
#endif
  Execute(context, &input, &output);
  EXPECT_EQ(output, input > input);

  // Destroy the engine.
  context->destroy();
  engine->destroy();
  runtime->destroy();
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
