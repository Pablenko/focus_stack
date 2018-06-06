#include <experimental/filesystem>

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

std::vector<cv::Mat> read_imgs(const std::string& path)
{
	namespace fs = std::experimental::filesystem;

	std::vector<cv::Mat> imgs;

	for(auto & f : fs::directory_iterator(path))
	{
		auto item_path = f.path().string();

		if(is_image(item_path))
		{
			imgs.push_back(cv::imread(item_path));
		}
	}

	return imgs;
}

cv::Mat merge_images(const cv::Mat& first, const cv::Mat& second)
{
	cv::Mat result(first.size().height, first.size().width + second.size().width, CV_8UC3);
    cv::Mat left(result, cv::Rect(0, 0, first.size().width, first.size().height));
    first.copyTo(left);
    cv::Mat right(result, cv::Rect(first.size().width, 0, second.size().width, second.size().height));
    second.copyTo(right);

    return result;
}

std::tuple<unsigned int, unsigned int> get_merged_size(cv::Size size, unsigned int scaler)
{
	float w = size.width;
	float h = size.height;
	float prop = w/h;
	return {2 * prop * scaler, scaler};
}
