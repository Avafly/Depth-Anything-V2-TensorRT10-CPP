#include "dptv2.h"

#include "3rdparty/CLI11.hpp"
#include "3rdparty/spdlog/common.h"
#include "3rdparty/spdlog/spdlog.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

const std::vector<std::string> image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};

bool IsImageFile(const std::filesystem::path &path)
{
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return std::find(image_exts.begin(), image_exts.end(), ext) != image_exts.end();
}

std::vector<std::filesystem::path> CollectImagePaths(const std::filesystem::path &dir)
{
    std::vector<std::filesystem::path> paths;
    for (const auto &entry : std::filesystem::directory_iterator(dir))
    {
        if (entry.is_regular_file() && IsImageFile(entry.path()))
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<cv::Mat> LoadImages(const std::vector<std::filesystem::path> &paths)
{
    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    for (const auto &path : paths)
    {
        cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR_BGR);
        if (img.empty())
            SPDLOG_WARN("Failed to load image: {}", path.string());
        else
            images.push_back(std::move(img));
    }
    return images;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        CLI::App app{};

        std::string model_path;
        std::string input_path;
        std::string output_dir;
        int target_size;
        int batch_size;
        int warmup;

        app.add_option("-m,--model", model_path, "Model path");
        app.add_option("-i,--input", input_path, "Image or folder path");
        app.add_option("-o,--output", output_dir, "Output directory")->default_val(".");
        app.add_option("-s,--size", target_size, "Target size")->default_val(518);
        app.add_option("-b,--batch", batch_size, "Max batch size")->default_val(0);
        app.add_option("-w,--warmup", warmup, "Warmup rounds")->default_val(0);

        CLI11_PARSE(app, argc, argv);

        if (model_path.empty() || input_path.empty())
        {
            std::fprintf(stderr, "%s", app.help().c_str());
            return 1;
        }

        // init spdlog
        spdlog::set_pattern("[%^%L%$] [%s:%#] %v");
        spdlog::set_level(spdlog::level::debug);
        spdlog::flush_on(spdlog::level::err);

        SPDLOG_INFO("Model: {}", model_path);
        SPDLOG_INFO("Input: {}", input_path);
        SPDLOG_INFO("Target size: {}", target_size);

        std::filesystem::create_directories(output_dir);

        // create model
        Infer::DPTv2 dpt(model_path, target_size, batch_size);

        if (warmup > 0)
            dpt.Warmup(warmup);

        if (std::filesystem::is_directory(input_path))
        {
            // batch inference if input is a folder path
            auto image_paths = CollectImagePaths(input_path);
            if (image_paths.empty())
                throw std::runtime_error("No images found in: " + input_path);

            const int max_batch_size = dpt.GetMaxBatchSize();
            const int total = static_cast<int>(image_paths.size());
            SPDLOG_INFO("Found {} images, processing in batches of {}", total, max_batch_size);

            double total_time = 0.0;
            int saved_count = 0;
            for (int offset = 0; offset < total; offset += max_batch_size)
            {
                int count = std::min(max_batch_size, total - offset);

                std::vector<std::filesystem::path> batch_paths(
                    image_paths.begin() + offset, image_paths.begin() + offset + count);
                auto images = LoadImages(batch_paths);
                if (images.empty())
                    continue;

                // predict and save
                auto start_time = cv::getTickCount();
                auto results = dpt.Predict(images);
                total_time += (cv::getTickCount() - start_time) / cv::getTickFrequency() * 1000.0;
                for (const auto &result : results)
                {
                    std::filesystem::path out_path = std::filesystem::path(output_dir) /
                                                     (std::to_string(saved_count++) + "_depthb.png");
                    if (!cv::imwrite(out_path.string(), result))
                        SPDLOG_WARN("Failed to save: {}", out_path.string());
                }

                SPDLOG_INFO("Processed batch {}/{} ({} images)", offset / max_batch_size + 1,
                            (total + max_batch_size - 1) / max_batch_size, count);
            }
            SPDLOG_DEBUG("Total time: {:.1f} ms ({} images)", total_time, saved_count);
        }
        else
        {
            // single image mode
            cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR_BGR);
            if (image.empty())
                throw std::runtime_error("Failed to load image: " + input_path);

            [[maybe_unused]] auto start_time = cv::getTickCount();
            auto result = dpt.Predict(image);
            SPDLOG_DEBUG("Total time: {:.1f} ms",
                         (cv::getTickCount() - start_time) / cv::getTickFrequency() * 1000.0);
            if (!result.empty())
            {
                std::filesystem::path output_path =
                    std::filesystem::path(output_dir) /
                    (std::filesystem::path(input_path).stem().string() + "_depth.png");
                if (!cv::imwrite(output_path.string(), result))
                    SPDLOG_WARN("Failed to save: {}", output_path.string());
                else
                    SPDLOG_INFO("Result image saved");
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        SPDLOG_ERROR("Error: {}", e.what());
        spdlog::shutdown();
        return 1;
    }
}