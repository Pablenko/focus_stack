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

cv::Mat focus_stack_average_method(const std::vector<cv::Mat>& in)
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

unsigned char get_kernel_sum_n_chan(const cv::Mat& in, const kernel<3>& k, unsigned int channel_num, unsigned int h, unsigned int w)
{
    unsigned char result = 0;

    for(int i = 0, w_offset = -1; i < 3; i++, w_offset++)
    {
        for(int j = 0, h_offset = -1; j < 3; j++, h_offset++)
        {
            float kernel_weigth = k(i, j);
            unsigned char val = in.at<cv::Vec3b>(h+h_offset, w+w_offset)[channel_num];
            result += kernel_weigth * val;
        }
    }

    return result;
}

short int get_kernel_sum(const cv::Mat& in, const kernel<3>& k, unsigned int h, unsigned int w)
{
    short int result = 0;

    for(int i = 0, w_offset = -1; i < 3; i++, w_offset++)
    {
        for(int j = 0, h_offset = -1; j < 3; j++, h_offset++)
        {
            float kernel_weigth = k(i, j);
            short int val = in.at<short int>(h+h_offset, w+w_offset);
            result += kernel_weigth * val;
        }
    }

    return result;
}

void apply_kernel_3_3(cv::Mat& in, const kernel<3>& k)
{
    cv::Size in_size = in.size();
    const unsigned int offset = 1;
    cv::Mat calc = in.clone();

    for(auto i = offset; i < in_size.height - offset; i++)
    {
        for(auto j = offset; j < in_size.width - offset; j++)
        {
            in.at<cv::Vec3b>(i, j)[0] = get_kernel_sum_n_chan(calc, k, 0, i, j);
            in.at<cv::Vec3b>(i, j)[1] = get_kernel_sum_n_chan(calc, k, 1, i, j);
            in.at<cv::Vec3b>(i, j)[2] = get_kernel_sum_n_chan(calc, k, 2, i, j);
        }
    }
}

void apply_kernel_3_1(cv::Mat& in, const kernel<3>& k)
{
    cv::Size in_size = in.size();
    const unsigned int offset = 1;
    cv::Mat calc = in.clone();

    for(auto i = offset; i < in_size.height - offset; i++)
    {
        for(auto j = offset; j < in_size.width - offset; j++)
        {
            in.at<short int>(i, j) = get_kernel_sum(calc, k, i, j);
        }
    }
}

static void gaussian_blur(cv::Mat& in)
{
    const unsigned int kernel_size = 3;

    kernel<kernel_size> gaussian_kernel = {0.077847, 0.123317, 0.077847, 0.123317, 0.195346, 0.123317, 0.077847, 0.123317, 0.077847};

    apply_kernel_3_3(in, gaussian_kernel);
}

static cv::Mat rgb_2_grayscale(const cv::Mat& in, int type)
{
    cv::Size size = in.size();
    cv::Mat result(size, type);
    const float b_weight = 0.11;
    const float g_weight = 0.59;
    const float r_weight = 0.3;

    for(auto i = 0; i < size.height; i++)
    {
        for(auto j = 0; j < size.width; j++)
        {
            short int gray_value = b_weight * in.at<cv::Vec3b>(i, j)[0] +
                                   g_weight * in.at<cv::Vec3b>(i, j)[1] +
                                   r_weight * in.at<cv::Vec3b>(i, j)[2];
            result.at<short int>(i, j) = gray_value;
        }
    }

    return result;
}

static void laplacian(cv::Mat& in)
{
    const unsigned int kernel_size = 3;

    kernel<kernel_size> laplacian_kernel = {0, 1, 0, 1, -4, 1, 0, 1, 0};

    apply_kernel_3_1(in, laplacian_kernel);
}

static void mat_abs(cv::Mat& in)
{
    cv::Size size = in.size();
    auto abs = [](short int i) -> short int { if(i < 0) {return -i;} else {return i;} };

    for(auto i = 0; i < size.height; i++)
    {
        for(auto j = 0; j < size.width; j++)
        {
            in.at<short int>(i, j) = abs(in.at<short int>(i, j));
        }
    }
}

static unsigned int get_max_pixel_index(const std::vector<cv::Mat>& edges, unsigned int h, unsigned int w)
{
    short int max_value = 0, idx = 0;

    for(unsigned int i = 0; i < edges.size(); i++)
    {
        auto cur_value = edges[i].at<short int>(h, w);
        if(cur_value > max_value)
        {
            max_value = cur_value;
            idx = i;
        }
    }

    return idx;
}

cv::Mat convert(const cv::Mat& in)
{
    cv::Mat r(in.size(), CV_8UC1);

    for(int i = 0; i < in.size().height; i++)
    {
        for(int j =0; j < in.size().width; j++)
        {
            r.at<unsigned char>(i, j) = in.at<short int>(i, j);
        }
    }

    return r;
}

std::vector<cv::Mat> detect_edges(const std::vector<cv::Mat>& in)
{
    std::vector<cv::Mat> edges;

    for(const auto& elem : in)
    {
        cv::Mat m = elem.clone();
        gaussian_blur(m);
        cv::Mat gray_scale = rgb_2_grayscale(m, CV_16SC1);
        laplacian(gray_scale);
        mat_abs(gray_scale);
        edges.push_back(gray_scale);
    }

    return edges;
}

cv::Mat focus_stack_laplacian(const std::vector<cv::Mat>& in, const std::vector<cv::Mat>& edges)
{
    cv::Size size = in[0].size();
    cv::Mat result(size, CV_8UC3);

    for(auto i = 0; i < size.height; i++)
    {
        for(auto j = 0; j < size.width; j++)
        {
            auto idx = get_max_pixel_index(edges, i, j);
            result.at<cv::Vec3b>(i, j) = in[idx].at<cv::Vec3b>(i, j);
        }
    }

    return result;
}

cv::Mat depth_map(const std::vector<cv::Mat>& edges)
{
    cv::Size size = edges[0].size();
    cv::Mat result(size, CV_8UC1, cv::Scalar(0));
    const int threshold = 12;
    int scaler = edges.size() * 2;

    for(auto i = 0; i < edges.size(); i++, scaler -= 2)
    {
        for(auto h = 0; h < size.height; h++)
        {
            for(auto w = 0; w < size.width; w++)
            {
                if(edges[i].at<short int>(h, w) > threshold)
                {
                    result.at<unsigned char>(h, w) = edges[i].at<short int>(h, w) * scaler;
                }
            }
        }
    }

    return result;
}
