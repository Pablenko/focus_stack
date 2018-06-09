#ifndef IMG_ALGS_HPP
#define IMG_ALGS_HPP

#include <vector>

#include "opencv/cv.hpp"

template<unsigned int size>
struct kernel
{
    float params[size][size];

    float operator()(int i, int j) const
    {
        return params[i][j];
    }
};

void apply_kernel_3_3(cv::Mat& in, const kernel<3>& k);

void apply_kernel_3_1(cv::Mat& in, const kernel<3>& k);

short int get_kernel_sum(const cv::Mat& in, const kernel<3>& k, unsigned int h, unsigned int w);

cv::Mat focus_stack_average_method(const std::vector<cv::Mat>& in);

cv::Mat focus_stack_laplacian(const std::vector<cv::Mat>& in);

cv::Mat depth_map_grayscale(const std::vector<cv::Mat>& in);

#endif
