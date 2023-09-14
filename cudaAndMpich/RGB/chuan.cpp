#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    cv::Mat image;
    image = cv::imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::Mat gray_image(image.rows, image.cols, CV_8UC1);

    auto start = std::chrono::high_resolution_clock::now();
    
    for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
            cv::Vec3b rgbPixel = image.at<cv::Vec3b>(y, x);
            gray_image.at<uchar>(y,x) = 0.2989*rgbPixel[2] + 0.5870*rgbPixel[1] + 0.1140*rgbPixel[0]; // convert RGB to grayscale
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::string output_name = std::string(argv[1]) + "_cpu_output.jpg";
    cv::imwrite(output_name, gray_image);

    std::chrono::duration<double, std::milli> diff = end-start;
    std::ofstream outfile("RGBtime.txt", std::ios_base::app | std::ios_base::out);
    outfile << output_name << " 用时： " << diff.count() << " ms" << std::endl;

    return 0;
}
