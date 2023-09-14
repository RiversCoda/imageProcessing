#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

__global__ void matrix_multiplication(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 线程所在的行
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 线程所在的列

    if (row < n && col < n) { // 只有在矩阵维度内的线程才进行计算
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void multiply_matrices(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int n)
{
    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    // cuda copy and counting time
    auto stime1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // 此处的思路是开更大的矩阵以满足不可整除的情况，在kernel中判断是否越界
    dim3 block_size(16, 16);                                                                      // 单个block中的线程数
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); // block的数量 = (矩阵维度 + block_size - 1) / block_size

    auto start_time = std::chrono::high_resolution_clock::now();
    matrix_multiplication<<<grid_size, block_size>>>(d_A, d_B, d_C, n); // kernel函数
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // std::cout << "Duration: " << duration << " µs" << std::endl;

    std::ofstream times_file("times_gpu.txt", std::ios::app);
    if (times_file) {
        times_file << "gpu " << n << " " << duration << " µs" << std::endl;
        times_file.close();
    }
    else {
        std::cout << "Unable to open times.txt file." << std::endl;
    }

    cudaMemcpy(C.data(), d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    auto etime1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(etime1 - stime1).count();
    std::ofstream times_file1("times_gpu_include_memcpy_time.txt", std::ios::app);
    if (times_file1) {
        times_file1 << "gpu " << n << " " << duration1 << " µs" << std::endl;
        times_file1.close();
    }
    else {
        std::cout << "Unable to open times.txt file." << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    std::vector<int> sizes = {100, 200, 500, 1000, 1500, 2000, 3000, 5000};

    std::ofstream times_file("times.txt");

    if (times_file) {
        for (int n : sizes) {
            std::vector<float> A(n * n);
            std::vector<float> B(n * n);
            std::vector<float> C(n * n);

            // Generate random matrices A and B
            for (int i = 0; i < n * n; ++i) {
                // static_cast<float> 是强制类型转换
                // rand() 产生一个随机数
                // RAND_MAX 是随机数的最大值
                // 给A[i]和B[i]赋值为0到1之间的随机数
                A[i] = static_cast<float>(rand()) / RAND_MAX;
                B[i] = static_cast<float>(rand()) / RAND_MAX;
            }

            multiply_matrices(A, B, C, n);

            std::ofstream file("C_gpu_" + std::to_string(n) + ".bin", std::ios::binary);
            if (file) {
                // reinterpret_cast 是强制类型转换
                // reinterpret_cast 和 static_cast 的区别是，reinterpret_cast 可以将任何指针转换为任何其他指针类型，而 static_cast 只能用于相关类型的转换
                file.write(reinterpret_cast<char *>(C.data()), C.size() * sizeof(float));
                file.close();
            }
        }

        times_file.close();
    }
    else {
        std::cout << "Unable to open times.txt file." << std::endl;
    }

    return 0;
}
