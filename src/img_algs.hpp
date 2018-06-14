#ifndef IMG_ALGS_HPP
#define IMG_ALGS_HPP

#include <vector>

#include "opencv/cv.hpp"

cv::Mat rgb_2_grayscale(const cv::Mat& in, int type);

void gaussian_blur(cv::Mat& in);

void laplacian(cv::Mat& in);

std::vector<cv::Mat> detect_edges(const std::vector<cv::Mat>& in);

cv::Mat focus_stack_laplacian(const std::vector<cv::Mat>& in, const std::vector<cv::Mat>& edges);

cv::Mat depth_map(const std::vector<cv::Mat>& in);

#endif
