#include "img_algs.hpp"
#include <iostream>

static cv::Vec3b average(const std::vector<cv::Mat>& in, unsigned int i, unsigned int j)
{
	unsigned int a1 = 0;
	unsigned int a2 = 0;
	unsigned int a3 = 0;

	for(const auto& e : in)
	{
		a1 += e.at<cv::Vec3b>(i, j)[0];
		a2 += e.at<cv::Vec3b>(i, j)[1];
		a3 += e.at<cv::Vec3b>(i, j)[2];
	}

	a1 /= in.size();
	a2 /= in.size();
	a3 /= in.size();

	return {static_cast<unsigned char>(a1), static_cast<unsigned char>(a2), static_cast<unsigned char>(a3)};
}

cv::Mat focus_stack(const std::vector<cv::Mat>& in)
{
	cv::Size elem_size = in[0].size(); // assume all images have same size
	cv::Mat result(elem_size, CV_8UC3);

    for(auto i = 0; i < elem_size.height; i++)
    {
        for(auto j = 0; j < elem_size.width; j++)
        {
            result.at<cv::Vec3b>(i, j) = average(in, i, j);
        }
    }

	return result;
}

cv::Mat depth_map_grayscale(const std::vector<cv::Mat>& in)
{
	return in[12];
}
