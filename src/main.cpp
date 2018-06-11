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

	std::vector<cv::Mat> edges = detect_edges(input);

	cv::Mat sharp = focus_stack_laplacian(input, edges);
	cv::Mat map = depth_map(edges);

	cv::namedWindow(window_name.c_str(), cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name.c_str(), window_w, window_h);

	cv::imshow(window_name.c_str(), sharp);
	cv::waitKey(0);

	cv::imshow(window_name.c_str(), map);
	cv::waitKey(0);
}
