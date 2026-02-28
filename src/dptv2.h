#ifndef DPTV2_H_
#define DPTV2_H_

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <driver_types.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <memory>
#include <string>
#include <vector>

namespace Infer
{

class DPTv2
{
public:
    explicit DPTv2(const std::string &model_path, const int target_size,
                   const int max_batch_size = 0);
    ~DPTv2();

    DPTv2(DPTv2 &&other) noexcept;
    DPTv2 &operator=(DPTv2 &&other) noexcept;

    DPTv2(const DPTv2 &) = delete;
    DPTv2 &operator=(const DPTv2 &) = delete;

    cv::Mat Predict(const cv::Mat &image_bgr) const;
    std::vector<cv::Mat> Predict(const std::vector<cv::Mat> &images_bgr) const;

    void Warmup(int rounds = 3);
    int GetMaxBatchSize() const noexcept;

private:
    int target_size_{};
    int max_batch_size_{};
    std::string in_name_{"image"};
    std::string out_name_{"depth"};

    std::unique_ptr<cudaStream_t> stream_{};
    std::unique_ptr<nvinfer1::IRuntime> runtime_{};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{};

    std::vector<void *> buffers_{};
    float *pinned_in_host_{};
    float *pinned_out_host_{};

    static inline const cv::Scalar mean_bgr_{0.406, 0.456, 0.485};
    static inline const cv::Scalar norm_bgr_{0.225, 0.224, 0.229};

    std::vector<char> BuildEngineFromOnnx(const std::string &onnx_path);
    void Cleanup() noexcept;
};

inline int DPTv2::GetMaxBatchSize() const noexcept
{
    return max_batch_size_;
}

} // namespace Infer

#endif // DPTV2_H_