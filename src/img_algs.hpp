#ifndef IMG_ALGS_HPP
#define IMG_ALGS_HPP

#include <vector>

#include "opencv/cv.hpp"

cv::Mat focus_stack(const std::vector<cv::Mat>& in);

cv::Mat depth_map_grayscale(const std::vector<cv::Mat>& in);

#endif
