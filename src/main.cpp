#include <iostream>
#include <string>

#include "opencv/cv.hpp"

#include "common.hpp"
#include "img_algs.hpp"

int main(int argc, char** argv)
{
    const std::string window_name_sharp = "Sharp version of image";
    const std::string window_name_depth_map = "Depth map of image";
    const std::string path = argv[argc-1];

    std::vector<cv::Mat> input = read_imgs(path);

    if(input.size() == 0)
    {
        std::cerr << "NO IMAGES FOUND!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> edges = detect_edges(input);

    cv::Mat sharp = focus_stack_laplacian(input, edges);
    cv::Mat map = depth_map(edges);

    const float scaler = 0.7; // Set one you wish
    auto [window_w, window_h] = get_window_size(sharp.size(), scaler);

    cv::namedWindow(window_name_sharp.c_str(), cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_sharp.c_str(), window_w, window_h);

    cv::imshow(window_name_sharp.c_str(), sharp);
    cv::waitKey(0);

    cv::namedWindow(window_name_depth_map.c_str(), cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_depth_map.c_str(), window_w, window_h);

    cv::imshow(window_name_depth_map.c_str(), map);
    cv::waitKey(0);

    save_images(path, sharp, map);
}
