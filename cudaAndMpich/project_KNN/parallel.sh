nvcc --gpu-architecture=sm_37 parallel.cu -o parallel
CUDA_VISIBLE_DEVICES=2 ./parallel

