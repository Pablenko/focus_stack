#include <iostream>
#include <string>

#include "opencv/cv.hpp"

#include "common.hpp"
#include "img_algs.hpp"

int main(int argc, char** argv)
{
	const std::string window_name = "Image magic";
	const std::string path = argv[argc-1];

	std::vector<cv::Mat> input = read_imgs(path);

	if(input.size() == 0)
	{
		std::cerr << "NO IMAGES FOUND!" << std::endl;
		return -1;
	}

    const unsigned int scaler = 600; // Set one you wish
	auto [window_w, window_h] = get_merged_size(input[0].size(), scaler);

	cv::Mat sharp = focus_stack(input);
	cv::Mat gray_depth_scale = depth_map_grayscale(input);
    cv::Mat merged_output = merge_images(sharp, gray_depth_scale);

	cv::namedWindow(window_name.c_str(), cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name.c_str(), window_w, window_h);

	cv::imshow(window_name.c_str(), merged_output);

	cv::waitKey(0);
}
