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

    cv::Mat gray_image;
    auto start = std::chrono::high_resolution_clock::now();
    cv::cvtColor( image, gray_image, cv::COLOR_BGR2GRAY );
    auto end = std::chrono::high_resolution_clock::now();

    std::string output_name = std::string(argv[1]) + "_cpu_output_copy.jpg";
    cv::imwrite(output_name, gray_image);

    std::chrono::duration<double, std::milli> diff = end-start;
    std::ofstream outfile("RGBtime.txt", std::ios_base::app | std::ios_base::out);
    outfile << output_name << " 用时： " << diff.count() << " ms" << std::endl;

    return 0;
}
