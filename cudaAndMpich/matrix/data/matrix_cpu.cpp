#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

void multiply_matrices(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    std::vector<int> sizes = {100, 200, 500, 1000, 1500, 2000, 3000, 5000};

    std::ofstream times_file("times.txt");

    if (times_file) {
        for (int n : sizes) {
            std::vector<float> A(n * n);
            std::vector<float> B(n * n);
            std::vector<float> C(n * n);

            // Generate random matrices A and B
            for (int i = 0; i < n * n; ++i) {
                A[i] = static_cast<float>(rand()) / RAND_MAX;
                B[i] = static_cast<float>(rand()) / RAND_MAX;
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            multiply_matrices(A, B, C, n);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            times_file << "cpu " << n << " " << duration << " ms" << std::endl;

            std::ofstream file("C_" + std::to_string(n) + ".bin", std::ios::binary);
            if (file) {
                file.write(reinterpret_cast<char*>(C.data()), C.size() * sizeof(float));
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
