
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <chrono>
#include <fstream>

#define CHANNELS 3

using namespace cv;

__global__ void colorToGrayscale(unsigned char *rgb, unsigned char *gray, int width, int height) {// 传入rgb图像和灰度图像的指针，以及图像的宽和高
    // 计算当前线程的坐标
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // 如果坐标在图像范围内
    if (x < width && y < height) {
        // 计算灰度图像的偏移量
        int grayOffset = y*width + x;
        // 计算rgb图像的偏移量
        int rgbOffset = grayOffset*CHANNELS;
        // 计算灰度值
        unsigned char r = rgb[rgbOffset];
        unsigned char g = rgb[rgbOffset + 1];
        unsigned char b = rgb[rgbOffset + 2];
        // 将灰度值写入灰度图像
        gray[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

int main(int argc, char **argv) {
    char *imageName = argv[1];
    // 使用opencv的Mat类存储图像，Mat是一个矩阵类，可以存储多维数组
    Mat image;
    image = imread(imageName, 1);
    // 抛出异常
    if (argc != 2 || !image.data) {
        printf("No image data \n");
        return -1;
    }

    int imageSize = image.rows*image.cols*CHANNELS;
    int grayImageSize = image.rows*image.cols;
    // 分配内存
    unsigned char *d_image, *d_grayImage, *h_image, *h_grayImage;

    h_image = (unsigned char*)malloc(imageSize);
    h_image = image.ptr();

    h_grayImage = (unsigned char*)malloc(grayImageSize);
    // 将图像数据从主机内存复制到设备内存
    cudaMalloc((void**)&d_image, imageSize);
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_grayImage, grayImageSize);
    // 定义线程块和线程网格的大小
    dim3 dimBlock(32, 32);
    dim3 dimGrid(image.cols/dimBlock.x, image.rows/dimBlock.y);

    auto start = std::chrono::high_resolution_clock::now();

    colorToGrayscale<<<dimGrid, dimBlock>>>(d_image, d_grayImage, image.cols, image.rows);

    cudaMemcpy(h_grayImage, d_grayImage, grayImageSize, cudaMemcpyDeviceToHost);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::string outputFileName = "gpu_output.jpg";
    std::string timeLogFileName = "RGBtime.txt";
    // 将灰度图像数据从设备内存复制到主机内存
    Mat gray_image;
    gray_image.create(image.rows, image.cols, CV_8UC1);
    memcpy(gray_image.ptr(), h_grayImage, grayImageSize);
    imwrite(outputFileName, gray_image);

    // 将运行时间写入日志文件
    std::ofstream logFile;
    logFile.open(timeLogFileName, std::ios_base::app);
    logFile << outputFileName << " 用时： " << duration.count() << " ms\n";
    logFile.close();

    cudaFree(d_image);
    cudaFree(d_grayImage);

    return 0;
}