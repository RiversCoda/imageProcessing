#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>

// function to calculate matrix multiplication
void matrix_mult(int n, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n+j] = 0;
            for (int k = 0; k < n; k++) {
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> sizes = {100, 200, 500, 1000, 1500, 2000, 3000, 5000};

    if (rank == 0) {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0,1.0);

        // master process
        for (auto& n : sizes) {
            std::vector<float> A(n * n);
            std::vector<float> B(n * n);
            std::vector<float> C(n * n);

            // Fill the matrices A and B with random float values.
            for (int i = 0; i < n * n; ++i) {
                A[i] = distribution(generator);
                B[i] = distribution(generator);
            }
            auto start_time = std::chrono::high_resolution_clock::now();
            matrix_mult(n, A, B, C);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::ofstream time_file("C_mpi_time.txt", std::ios_base::app);
            if(time_file.is_open()) {
                time_file << n << " " << duration.count() << "\n";
                time_file.close();
            }
            else {
                std::cout << "Unable to open file\n";
            }
            

            // write to file
            std::string filename = "C_mpi_size_" + std::to_string(n) + ".txt";
            std::ofstream f(filename);
            if (f.is_open()) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        f << C[i*n+j] << " ";
                    }
                    f << "\n";
                }
                f.close();
            }
            else {
                std::cout << "Unable to open file\n";
            }
        }
    }

    MPI_Finalize();
}
