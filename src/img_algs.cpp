#include <cmath>
#include <iostream>

#include "img_algs.hpp"

namespace 
{

struct kernel
{
    static const unsigned int size = 3;
    float params[size][size];
 
    float operator()(int i, int j) const
    {
        return params[i][j];
    }
};

unsigned char get_kernel_sum_n_chan(const cv::Mat& in, const kernel& k, unsigned int channel_num, unsigned int h, unsigned int w)
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

short int get_kernel_sum(const cv::Mat& in, const kernel& k, unsigned int h, unsigned int w)
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

void apply_kernel_3_ch(cv::Mat& in, const kernel& k)
{
    const cv::Size in_size = in.size();
    const int offset = 1;
    const cv::Mat calc = in.clone();

    for(int i = offset; i < in_size.height - offset; i++)
    {
        for(int j = offset; j < in_size.width - offset; j++)
        {
            in.at<cv::Vec3b>(i, j)[0] = get_kernel_sum_n_chan(calc, k, 0, i, j);
            in.at<cv::Vec3b>(i, j)[1] = get_kernel_sum_n_chan(calc, k, 1, i, j);
            in.at<cv::Vec3b>(i, j)[2] = get_kernel_sum_n_chan(calc, k, 2, i, j);
        }
    }
}

void apply_kernel(cv::Mat& in, const kernel& k)
{   
    const int offset = 1;
    const cv::Size in_size = in.size();
    const cv::Mat calc = in.clone();

    for(int i = offset; i < in_size.height - offset; i++)
    {
        for(int j = offset; j < in_size.width - offset; j++)
        {
            in.at<short int>(i, j) = get_kernel_sum(calc, k, i, j);
        }
    }
}

void mat_abs(cv::Mat& in)
{
    const cv::Size size = in.size();

    for(int i = 0; i < size.height; i++)
    {
        for(int j = 0; j < size.width; j++)
        {
            in.at<short int>(i, j) = std::abs(in.at<short int>(i, j));
        }
    }
}

int get_max_index(const std::vector<cv::Mat>& edges, int h, int w)
{
    int max_value = 0, idx = 0;

    for(unsigned int i = 0; i < edges.size(); i++)
    {
        int cur_value = edges[i].at<short int>(h, w);
        if(cur_value > max_value)
        {
            max_value = cur_value;
            idx = i;
        }
    }

    return idx;
}

int get_max_block_index(const std::vector<cv::Mat>& edges, int h, int w)
{
    const kernel sum_kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    int max_value = 0, idx = 0;

    for(unsigned int i = 0; i < edges.size(); i++)
    {
        int cur_value = get_kernel_sum(edges[i], sum_kernel, h, w);
        if(cur_value > max_value)
        {
            max_value = cur_value;
            idx = i;
        }
    }

    return idx;
}

void fill_borders(const std::vector<cv::Mat>& in, const std::vector<cv::Mat>& edges, cv::Mat& result)
{
    const cv::Size size = result.size();

    for(int i = 0; i < size.height; i+=size.height-1)
    {
        for(int j = 0; j < size.width; j++)
        {
            int idx = get_max_index(edges, i, j);
            result.at<cv::Vec3b>(i, j) = in[idx].at<cv::Vec3b>(i, j);
        }
    }

    for(int i = 0; i < size.width; i+=size.width-1)
    {
        for(int j = 0; j < size.height; j++)
        {
            int idx = get_max_index(edges, j, i);
            result.at<cv::Vec3b>(j, i) = in[idx].at<cv::Vec3b>(j, i);
        }
    }
}

} //unnamed namespace

cv::Mat rgb_2_grayscale(const cv::Mat& in, int type)
{  
    const float b_weight = 0.11;
    const float g_weight = 0.59;
    const float r_weight = 0.3;
    const cv::Size size = in.size();

    cv::Mat result(size, type);

    for(int i = 0; i < size.height; i++)
    {
        for(int j = 0; j < size.width; j++)
        {
            short int gray_value = b_weight * in.at<cv::Vec3b>(i, j)[0] +
                                   g_weight * in.at<cv::Vec3b>(i, j)[1] +
                                   r_weight * in.at<cv::Vec3b>(i, j)[2];
            result.at<short int>(i, j) = gray_value;
        }
    }

    return result;
}


void gaussian_blur(cv::Mat& in)
{
    const kernel gaussian_kernel = {0.077847, 0.123317, 0.077847,
                                    0.123317, 0.195346, 0.123317,
                                    0.077847, 0.123317, 0.077847};

    apply_kernel_3_ch(in, gaussian_kernel);
}

void laplacian(cv::Mat& in)
{
    const kernel laplacian_kernel = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

    apply_kernel(in, laplacian_kernel);
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
    const cv::Size size = in[0].size();

    cv::Mat result(size, CV_8UC3, cv::Scalar(0, 0, 0));

    fill_borders(in, edges, result);

    for(int i = 1; i < size.height - 1; i+=1)
    {
        for(int j = 1; j < size.width - 1; j+=1)
        {
            int idx = get_max_block_index(edges, i, j);
            result.at<cv::Vec3b>(i, j) = in[idx].at<cv::Vec3b>(i, j);
        }
    }

    return result;
}

cv::Mat depth_map(const std::vector<cv::Mat>& edges)
{
    const int detection_threshold = 22; // arbitrary chosen value to get rid of noise
    const int max_pixel_value = 255;
    const int min_pixel_value = 0;
    const cv::Size size = edges[0].size();

    cv::Mat result(size, CV_8UC1, cv::Scalar(0));
    int scale_step = (max_pixel_value - min_pixel_value) / edges.size();

    for(unsigned int i = 0, scaler = max_pixel_value - scale_step; i < edges.size(); i++, scaler -= scale_step)
    {
        for(int h = 0; h < size.height; h++)
        {
            for(int w = 0; w < size.width; w++)
            {
                if(edges[i].at<short int>(h, w) > detection_threshold)
                {
                    if(result.at<unsigned char>(h, w) == 0) // Dont overwrite earlier detection
                    {
                        result.at<unsigned char>(h, w) = static_cast<unsigned char>(scaler + (edges[i].at<short int>(h, w) / scale_step));
                    }
                }
            }
        }
    }

    return result;
}
