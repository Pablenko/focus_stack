#include <algorithm>
#include <experimental/filesystem>
#include <iostream>

#include "common.hpp"

static bool is_image(const std::string& loc)
{
    const std::vector<std::string> tags = {".jpg", ".png", ".bmp"};

    for(const auto& t : tags)
    {
        if(loc.find(t) != std::string::npos)
        {
            return true;
        }
    }

    return false;
}

static void summarize_read(const std::vector<std::string>& file_names)
{
    std::cout << "Found following images: " << std::endl;

    for(const auto& f: file_names)
    {
        std::cout << "    " << f << std::endl;
    }
}

std::vector<cv::Mat> read_imgs(const std::string& path)
{
    namespace fs = std::experimental::filesystem;

    std::vector<std::string> file_names;
    std::vector<cv::Mat> imgs;

    std::transform(fs::directory_iterator(path), fs::directory_iterator(), std::back_inserter(file_names), [](auto it) { return it.path().string();});
    file_names.erase(std::remove_if(file_names.begin(), file_names.end(), [](const std::string& loc) { return not is_image(loc);}), file_names.end());
    std::sort(file_names.begin(), file_names.end());

    for(auto& image_loc: file_names)
    {
        imgs.push_back(cv::imread(image_loc));
    }

    summarize_read(file_names);

    return imgs;
}

std::tuple<unsigned int, unsigned int> get_window_size(cv::Size size, float scaler)
{
    float h = size.height;
    float w = size.width;
    return {w * scaler, h * scaler};
}

static void summarize_write(const std::string& out_path_1, const std::string& out_path_2)
{
    std::cout << "Saved results to following locations:" << std::endl;
    std::cout << "    " << out_path_1 << std::endl;
    std::cout << "    " << out_path_2 << std::endl;
}

void save_images(const std::string& path, const cv::Mat& sharp, const cv::Mat& depth_map)
{
    namespace fs = std::experimental::filesystem;

    fs::path out_dir_path(path);
    out_dir_path /= "result";

    fs::path sharp_path(out_dir_path);
    sharp_path /= "sharp.png";

    fs::path depth_map_path(out_dir_path);
    depth_map_path /= "depth_map.png";

    fs::create_directory(out_dir_path);

    cv::imwrite(sharp_path.c_str(), sharp);
    cv::imwrite(depth_map_path.c_str(), depth_map);

    summarize_write(sharp_path.string(), depth_map_path.string());
}
