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
const char* kInput3Tensor = "input3";
const char* kInput4Tensor = "input4";
const char* kOutputTensor = "output";

// Creates a network to compute y=x>x.
nvinfer1::IHostMemory* CreateNetwork() {
  Logger logger;
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

  nvinfer1::INetworkDefinition* network = builder->createNetwork();
  auto add_input = [&](const char* name) {
    auto input = network->addInput(name, nvinfer1::DataType::kFLOAT,
                                 nvinfer1::DimsCHW{1, 1, 1});
    EXPECT_NE(input, nullptr);
    return input;
  };

  auto input1 = add_input(kInput1Tensor);
  auto input2 = add_input(kInput2Tensor);
  auto input3 = add_input(kInput3Tensor);
  auto input4 = add_input(kInput4Tensor);

  LOG(INFO) << "start comp";
  auto comp =
      network->addElementWise(*input1, *input2, ElementWiseOperation::kGREATER);
  EXPECT_NE(comp, nullptr);
  //comp->setOutputType(0, nvinfer1::DataType::kBOOL); // This line doesn't seem to be useful
  LOG(INFO) << "finish comp";

  auto cond = comp->getOutput(0);
  LOG(INFO) << "start select";
  auto select = network->addSelect(*cond, *input3, *input4);
  LOG(INFO) << "end select";

  auto output = select->getOutput(0);
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
void Execute(nvinfer1::IExecutionContext* context, const float* input1, const float* input2, const float* input3,const float* input4,
             float* output) {
  const nvinfer1::ICudaEngine& engine = context->getEngine();

  // We have two bindings: input and output.
  ASSERT_EQ(engine.getNbBindings(), 2);
  const int input1_index = engine.getBindingIndex(kInput1Tensor);
  ASSERT_EQ(0, input1_index);
  const int input2_index = engine.getBindingIndex(kInput2Tensor);
  ASSERT_EQ(1, input2_index);
  const int input3_index = engine.getBindingIndex(kInput3Tensor);
  ASSERT_EQ(2, input3_index);
  const int input4_index = engine.getBindingIndex(kInput4Tensor);
  ASSERT_EQ(3, input4_index);
  const int output_index = engine.getBindingIndex(kOutputTensor);
  ASSERT_EQ(4, output_index);

  // Create GPU buffers and a stream
  void* buffers[5];
  ASSERT_EQ(0, cudaMalloc(&buffers[input1_index], sizeof(float)));
  ASSERT_EQ(0, cudaMalloc(&buffers[input2_index], sizeof(float)));
  ASSERT_EQ(0, cudaMalloc(&buffers[input3_index], sizeof(float)));
  ASSERT_EQ(0, cudaMalloc(&buffers[input4_index], sizeof(float)));

  LOG(INFO) << "start cudaMalloc result";
  ASSERT_EQ(0, cudaMalloc(&buffers[output_index], sizeof(float)));
  LOG(INFO) << "finish cudaMalloc result";

  cudaStream_t stream;
  ASSERT_EQ(0, cudaStreamCreate(&stream));

  // Copy the input to the GPU, execute the network, and copy the output back.
  //
  // Note that since the host buffer was not created as pinned memory, these
  // async copies are turned into sync copies. So the following synchronization
  // could be removed.
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input1_index], input1, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input2_index], input2, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input3_index], input3, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(0, cudaMemcpyAsync(buffers[input4_index], input4, sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  context->enqueue(1, buffers, stream, nullptr);
  ASSERT_EQ(0, cudaMemcpyAsync(output, buffers[output_index], sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release the stream and the buffers
  cudaStreamDestroy(stream);
  ASSERT_EQ(0, cudaFree(buffers[input1_index]));
  ASSERT_EQ(0, cudaFree(buffers[input2_index]));
  ASSERT_EQ(0, cudaFree(buffers[input3_index]));
  ASSERT_EQ(0, cudaFree(buffers[input4_index]));
  ASSERT_EQ(0, cudaFree(buffers[output_index]));
}

TEST(TensorrtTest, SelectFP32) {
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
  float input1 = 12;
  float input2 = 16;
  float input3 = 11;
  float input4 = 17;
  float output;
  Execute(context, &input1, &input2, &input3, &input4, &output);
  EXPECT_EQ(output, input1 > input2 ? input3 : input4);

  // Destroy the engine.
  context->destroy();
  engine->destroy();
  runtime->destroy();
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
