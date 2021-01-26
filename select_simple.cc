/* reproducer to show a problem in creating a select layer using the result of a
 * logical greater layer as a condition.
 * NOTE: you probably need to adjust the include path, and put this file in
 * github tensorflow/compiler/tf2tensorrt/ test env to compile and run it.
 * But you can also just take the "core" of this code and put it on your test
 * env to compile and run.
 *
 error message:
W0126 11:24:22.208405    7293 select_simple.cc:109] TensorRT version 7.1.3
I0126 11:24:23.136196    7293 select_simple.cc:77] start comp
I0126 11:24:23.136241    7293 select_simple.cc:82] finish comp
I0126 11:24:23.136244    7293 select_simple.cc:85] start select
E0126 11:24:23.136254    7293 select_simple.cc:45] Parameter check failed at: ../builder/Network.cpp::addSelect::952, condition: !hasImplicitBatchDimension()
I0126 11:24:23.136279    7293 select_simple.cc:87] end select
*/

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
  comp->setOutputType(0, nvinfer1::DataType::kBOOL); // This line doesn't seem to be useful
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
  (void)model;
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
