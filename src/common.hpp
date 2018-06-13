#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <tuple>
#include <vector>

#include "opencv/cv.hpp"

std::vector<cv::Mat> read_imgs(const std::string& path);

std::tuple<unsigned int, unsigned int> get_window_size(cv::Size size, float scaler);

void save_images(const std::string& path, const cv::Mat& sharp, const cv::Mat& depth_map);

#endif
