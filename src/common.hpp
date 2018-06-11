#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <tuple>
#include <vector>

#include "opencv/cv.hpp"

std::vector<cv::Mat> read_imgs(const std::string& path);

cv::Mat merge_images(const cv::Mat& first, const cv::Mat& second, int type);

std::tuple<unsigned int, unsigned int> get_merged_size(cv::Size size, unsigned int scaler);

#endif
