#include "dptv2.h"

#include "logging.h"

#include "3rdparty/spdlog/spdlog.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#define CUDA_PRINT_ERROR(code)                                                                     \
    do                                                                                             \
    {                                                                                              \
        std::fprintf(stderr, "CUDA Error:\n");                                                     \
        std::fprintf(stderr, "    File:       %s\n", __FILE__);                                    \
        std::fprintf(stderr, "    Line:       %d\n", __LINE__);                                    \
        std::fprintf(stderr, "    Error code: %d\n", code);                                        \
        std::fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(code));                    \
    } while (0)

#define CUDA_ASSERT(call)                                                                          \
    do                                                                                             \
    {                                                                                              \
        const cudaError_t error_code = call;                                                       \
        if (error_code != cudaSuccess)                                                             \
        {                                                                                          \
            CUDA_PRINT_ERROR(error_code);                                                          \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#define CUDA_WARN(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        const cudaError_t error_code = call;                                                       \
        if (error_code != cudaSuccess)                                                             \
            CUDA_PRINT_ERROR(error_code);                                                          \
    } while (0)

namespace Infer
{

static Logger logger;

DPTv2::DPTv2(const std::string &model_path, const int target_size, const int max_batch_size)
    : target_size_(target_size)
    , max_batch_size_(max_batch_size)
{
    if (target_size_ < 14 || target_size_ % 14 != 0)
        throw std::runtime_error("Invalid target size: " + std::to_string(target_size_));

    // determine engine file path
    std::filesystem::path engine_path(model_path);

    if (engine_path.extension() == ".onnx")
    {
        engine_path.replace_extension(".engine");
        // build engine from onnx
        if (!std::filesystem::exists(engine_path))
        {
            SPDLOG_INFO("Converting engine from ONNX: {}", model_path);
            auto serialized = BuildEngineFromOnnx(model_path);

            // save to disk
            std::ofstream file(engine_path, std::ios::binary);
            if (!file)
                throw std::runtime_error("Failed to save engine to: " + engine_path.string());
            if (!file.write(serialized.data(), static_cast<std::streamsize>(serialized.size())))
                throw std::runtime_error("Failed to write engine data");
            SPDLOG_INFO("Saved engine to: {}", engine_path.string());
        }
        else
        {
            SPDLOG_INFO("Found cached engine: {}", engine_path.string());
        }
    }

    // load engine file
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file)
        throw std::runtime_error("Failed to open engine file: " + engine_path.string());
    engine_file.seekg(0, std::ios::end);
    std::streamsize engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    if (!engine_file.read(engine_data.data(), engine_size))
        throw std::runtime_error("Failed to read engine file");

    // create runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_)
        throw std::runtime_error("Failed to create runtime");

    // deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_)
        throw std::runtime_error("Failed to deserialize engine");

    // create context
    context_.reset(engine_->createExecutionContext());
    if (!context_)
        throw std::runtime_error("Failed to create context");

    // create stream
    stream_ = std::make_unique<cudaStream_t>();
    CUDA_ASSERT(cudaStreamCreate(stream_.get()));

    // get model info
    const int num_iotensors = engine_->getNbIOTensors();
    if (num_iotensors != 2)
        throw std::runtime_error("Excepted 2 IO tensors, got " + std::to_string(num_iotensors));
    for (int i = 0; i < num_iotensors; ++i)
    {
        const char *tensor_name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
            in_name_ = tensor_name;
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
            out_name_ = tensor_name;
    }
    // get max batch size
    int batch_size =
        engine_->getProfileShape(in_name_.c_str(), 0, nvinfer1::OptProfileSelector::kMAX).d[0];
    if (max_batch_size_ > batch_size)
        throw std::runtime_error("Invalid batch size: " + std::to_string(max_batch_size) +
                                 ", must be in [1, " + std::to_string(batch_size) + "]");
    if (max_batch_size_ <= 0)
        max_batch_size_ = batch_size;

    // create pinned host memory
    size_t max_in_size_byte = sizeof(float) * 3 * target_size_ * target_size_ * max_batch_size_;
    size_t max_out_size_byte = sizeof(float) * target_size_ * target_size_ * max_batch_size_;
    CUDA_ASSERT(cudaMallocHost(reinterpret_cast<void **>(&pinned_in_host_), max_in_size_byte));
    CUDA_ASSERT(cudaMallocHost(reinterpret_cast<void **>(&pinned_out_host_), max_out_size_byte));
    // create device memory
    buffers_.resize(num_iotensors);
    CUDA_ASSERT(cudaMalloc(&buffers_[0], max_in_size_byte));
    CUDA_ASSERT(cudaMalloc(&buffers_[1], max_out_size_byte));

    // set tensor addresses
    if (!context_->setInputTensorAddress(in_name_.c_str(), buffers_[0]) ||
        !context_->setOutputTensorAddress(out_name_.c_str(), buffers_[1]))
        throw std::runtime_error("Failed to set tensor addresses");
}

DPTv2::~DPTv2()
{
    Cleanup();
}

DPTv2::DPTv2(DPTv2 &&other) noexcept
    : target_size_(std::exchange(other.target_size_, {}))
    , max_batch_size_(std::exchange(other.max_batch_size_, {}))
    , in_name_(std::move(other.in_name_))
    , out_name_(std::move(other.out_name_))
    , stream_(std::move(other.stream_))
    , runtime_(std::move(other.runtime_))
    , engine_(std::move(other.engine_))
    , context_(std::move(other.context_))
    , buffers_(std::move(other.buffers_))
    , pinned_in_host_(std::exchange(other.pinned_in_host_, {}))
    , pinned_out_host_(std::exchange(other.pinned_out_host_, {}))
{
}

DPTv2 &DPTv2::operator=(DPTv2 &&other) noexcept
{
    if (this != &other)
    {
        Cleanup();

        target_size_ = std::exchange(other.target_size_, {});
        max_batch_size_ = std::exchange(other.max_batch_size_, {});
        in_name_ = std::move(other.in_name_);
        out_name_ = std::move(other.out_name_);
        stream_ = std::move(other.stream_);
        runtime_ = std::move(other.runtime_);
        engine_ = std::move(other.engine_);
        context_ = std::move(other.context_);
        buffers_ = std::move(other.buffers_);
        pinned_in_host_ = std::exchange(other.pinned_in_host_, {});
        pinned_out_host_ = std::exchange(other.pinned_out_host_, {});
    }
    return *this;
}

cv::Mat DPTv2::Predict(const cv::Mat &image_bgr) const
{
    // preprocessing
    int img_rows = image_bgr.rows;
    int img_cols = image_bgr.cols;
    float scale = static_cast<float>(target_size_) / std::max(img_rows, img_cols);
    int rsz_rows = static_cast<int>(img_rows * scale / 14.0) * 14;
    int rsz_cols = static_cast<int>(img_cols * scale / 14.0) * 14;

    cv::Mat processed;
    cv::resize(image_bgr, processed, cv::Size(rsz_cols, rsz_rows), 0, 0, cv::INTER_CUBIC);
    processed.convertTo(processed, CV_32FC3, 1.0 / 255.0);
    cv::subtract(processed, mean_bgr_, processed);
    cv::divide(processed, norm_bgr_, processed);
    const int blob_dims[4]{1, 3, rsz_rows, rsz_cols};
    cv::Mat blob(4, blob_dims, CV_32F, pinned_in_host_);
    cv::dnn::blobFromImage(processed, blob, 1.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    if (blob.data != reinterpret_cast<uchar *>(pinned_in_host_))
        throw std::runtime_error("blobFromImage did not write into pinned memory");

    // inference
    nvinfer1::Dims trt_in_dims{};
    trt_in_dims.nbDims = 4;
    trt_in_dims.d[0] = 1;
    trt_in_dims.d[1] = 3;
    trt_in_dims.d[2] = rsz_rows;
    trt_in_dims.d[3] = rsz_cols;
    if (!context_->setInputShape(in_name_.c_str(), trt_in_dims))
        throw std::runtime_error("Failed to set input shape");
    auto out_dims = context_->getTensorShape(out_name_.c_str());
    size_t in_size_byte = sizeof(float) * 3 * rsz_rows * rsz_cols;
    size_t out_size_byte = sizeof(float) * std::accumulate(out_dims.d, out_dims.d + out_dims.nbDims,
                                                           1, std::multiplies<int64_t>());

    SPDLOG_DEBUG("Input dims:  c={:<2} h={:<4} w={:<4}", trt_in_dims.d[1], trt_in_dims.d[2],
                 trt_in_dims.d[3]);
    SPDLOG_DEBUG("Output dims: c={:<2} h={:<4} w={:<4}", out_dims.d[0], out_dims.d[1],
                 out_dims.d[2]);

    // execute
    // measure pure inference elapsed time
    [[maybe_unused]] auto start_time = cv::getTickCount();

    CUDA_ASSERT(cudaMemcpyAsync(buffers_[0], pinned_in_host_, in_size_byte, cudaMemcpyHostToDevice,
                                *stream_));

    if (!context_->enqueueV3(*stream_))
        throw std::runtime_error("TensorRT inference failed");

    CUDA_ASSERT(cudaMemcpyAsync(pinned_out_host_, buffers_[1], out_size_byte,
                                cudaMemcpyDeviceToHost, *stream_));
    CUDA_ASSERT(cudaStreamSynchronize(*stream_));

    SPDLOG_DEBUG("Inference time: {:.1f} ms",
                 (cv::getTickCount() - start_time) / cv::getTickFrequency() * 1000.0);

    // postprocessing
    cv::Mat depth(out_dims.d[1], out_dims.d[2], CV_32FC1, pinned_out_host_);
    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::resize(depth, depth, cv::Size(img_cols, img_rows), 0, 0, cv::INTER_CUBIC);
    cv::applyColorMap(depth, depth, cv::COLORMAP_INFERNO);

    return depth;
}

std::vector<cv::Mat> DPTv2::Predict(const std::vector<cv::Mat> &images_bgr) const
{
    const int batch_size = static_cast<int>(images_bgr.size());
    if (batch_size == 0)
        return {};
    if (batch_size > max_batch_size_)
    {
        SPDLOG_ERROR("Batch size {} exceeds max batch size {}", batch_size, max_batch_size_);
        return {};
    }

    // compute unified resize dimensions
    int max_rows = 0, max_cols = 0;
    std::vector<cv::Size> original_sizes(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        original_sizes[i] = images_bgr[i].size();
        float scale =
            static_cast<float>(target_size_) / std::max(images_bgr[i].rows, images_bgr[i].cols);
        max_rows = std::max(max_rows, static_cast<int>(images_bgr[i].rows * scale / 14.0) * 14);
        max_cols = std::max(max_cols, static_cast<int>(images_bgr[i].cols * scale / 14.0) * 14);
    }

    // preprocessing
    std::vector<cv::Mat> processed_images(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        cv::resize(images_bgr[i], processed_images[i], cv::Size(max_cols, max_rows), 0, 0,
                   cv::INTER_CUBIC);
        processed_images[i].convertTo(processed_images[i], CV_32FC3, 1.0 / 255.0);
        cv::subtract(processed_images[i], mean_bgr_, processed_images[i]);
        cv::divide(processed_images[i], norm_bgr_, processed_images[i]);
    }
    const int blob_dims[4]{batch_size, 3, max_rows, max_cols};
    cv::Mat blob(4, blob_dims, CV_32F, pinned_in_host_);
    cv::dnn::blobFromImages(processed_images, blob, 1.0, cv::Size(), cv::Scalar(), true, false,
                            CV_32F);
    if (blob.data != reinterpret_cast<uchar *>(pinned_in_host_))
        throw std::runtime_error("blobFromImages did not write into pinned memory");

    // inference
    nvinfer1::Dims trt_in_dims{};
    trt_in_dims.nbDims = 4;
    trt_in_dims.d[0] = batch_size;
    trt_in_dims.d[1] = 3;
    trt_in_dims.d[2] = max_rows;
    trt_in_dims.d[3] = max_cols;
    if (!context_->setInputShape(in_name_.c_str(), trt_in_dims))
        throw std::runtime_error("Failed to set input shape");
    auto out_dims = context_->getTensorShape(out_name_.c_str());
    size_t in_size_byte = batch_size * 3 * max_rows * max_cols * sizeof(float);
    size_t out_size_byte =
        std::accumulate(out_dims.d, out_dims.d + out_dims.nbDims, 1, std::multiplies<int64_t>()) *
        sizeof(float);

    SPDLOG_DEBUG("Batch input dims:  b={} c={} h={} w={}", trt_in_dims.d[0], trt_in_dims.d[1],
                 trt_in_dims.d[2], trt_in_dims.d[3]);
    SPDLOG_DEBUG("Batch output dims: c={} h={} w={}", out_dims.d[0], out_dims.d[1], out_dims.d[2]);

    // execute
    [[maybe_unused]] auto start_time = cv::getTickCount();

    CUDA_ASSERT(cudaMemcpyAsync(buffers_[0], pinned_in_host_, in_size_byte, cudaMemcpyHostToDevice,
                                *stream_));

    if (!context_->enqueueV3(*stream_))
        throw std::runtime_error("TensorRT batch inference failed");

    CUDA_ASSERT(cudaMemcpyAsync(pinned_out_host_, buffers_[1], out_size_byte,
                                cudaMemcpyDeviceToHost, *stream_));
    CUDA_ASSERT(cudaStreamSynchronize(*stream_));

    SPDLOG_DEBUG("Batch inference time: {:.1f} ms ({} images)",
                 (cv::getTickCount() - start_time) / cv::getTickFrequency() * 1000.0, batch_size);

    // postprocessing
    std::vector<cv::Mat> results(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        cv::Mat depth(out_dims.d[1], out_dims.d[2], CV_32FC1,
                      pinned_out_host_ + i * max_rows * max_cols);
        cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::resize(depth, depth, original_sizes[i], 0, 0, cv::INTER_CUBIC);
        cv::applyColorMap(depth, results[i], cv::COLORMAP_INFERNO);
    }

    return results;
}

void DPTv2::Warmup(int rounds)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 4;
    dims.d[0] = 1;
    dims.d[1] = 3;
    dims.d[2] = target_size_;
    dims.d[3] = target_size_;
    if (!context_->setInputShape(in_name_.c_str(), dims))
        throw std::runtime_error("Failed to set input shape");

    for (int i = 0; i < rounds; ++i)
    {
        if (!context_->enqueueV3(*stream_))
            SPDLOG_WARN("Warmup inference failed at round {}", i);
        CUDA_ASSERT(cudaStreamSynchronize(*stream_));
    }
    SPDLOG_INFO("Warmup done ({} rounds)", rounds);
}

std::vector<char> DPTv2::BuildEngineFromOnnx(const std::string &onnx_path)
{
    // create builder
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
        throw std::runtime_error("Failed to create builder");

    // create network
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network)
        throw std::runtime_error("Failed to create network");

    // parse onnx
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser)
        throw std::runtime_error("Failed to create ONNX parser");

    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        for (int i = 0; i < parser->getNbErrors(); ++i)
            SPDLOG_ERROR("ONNX parse error: {}", parser->getError(i)->desc());
        throw std::runtime_error("Failed to parse ONNX model");
    }

    // create builder config
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config)
        throw std::runtime_error("Failed to create builder config");

    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // set dynamic shapes profile
    // matching: --minShapes=image:1x3x14x14
    //           --optShapes=image:1x3x518x518
    //           --maxShapes=image:2x3x1288x1288
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
    auto *input_tensor = network->getInput(0);
    if (!input_tensor)
        throw std::runtime_error("ONNX model has no input tensor");
    const char *input_name = input_tensor->getName();

    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4{1, 3, 14, 14});
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4{1, 3, 518, 518});
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4{2, 3, 1288, 1288});
    config->addOptimizationProfile(profile);

    SPDLOG_INFO("Building TensorRT engine (this may take a few minutes)...");

    // build serialized network
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized)
        throw std::runtime_error("Failed to build serialized network");

    std::vector<char> result(static_cast<const char *>(serialized->data()),
                             static_cast<const char *>(serialized->data()) + serialized->size());

    SPDLOG_INFO("Engine built successfully ({:.1f} MB)", result.size() / 1024.0f / 1024.0f);
    return result;
}

void DPTv2::Cleanup() noexcept
{
    for (const auto &buffer : buffers_)
        CUDA_WARN(cudaFree(buffer));
    buffers_.clear();

    CUDA_WARN(cudaFreeHost(pinned_in_host_));
    CUDA_WARN(cudaFreeHost(pinned_out_host_));
    pinned_in_host_ = nullptr;
    pinned_out_host_ = nullptr;

    if (stream_ && *stream_)
        CUDA_WARN(cudaStreamDestroy(*stream_));
    stream_.reset();
}

} // namespace Infer