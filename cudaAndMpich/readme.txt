注：考虑文件大小，大于等于1000*1000规模的数据已经删除，可以自行补充
matrix
├── data: cuda和串行所在文件夹
│   ├── cpl_gpu.sh: 矩阵乘法cuda程序编译脚本
│   ├── C_100.bin: 规模为100*100的cpu的矩阵输出结果
│   ├── C_1000.bin: 规模为1000*1000的cpu的矩阵输出结果
│   ├── C_1500.bin: 规模为1500*1500的cpu的矩阵输出结果
│   ├── C_200.bin: 规模为200*200的cpu的矩阵输出结果
│   ├── C_2000.bin: 规模为2000*2000的cpu的矩阵输出结果
│   ├── C_3000.bin: 规模为3000*3000的cpu的矩阵输出结果
│   ├── C_500.bin: 规模为500*500的cpu的矩阵输出结果
│   ├── C_5000.bin: 规模为5000*5000的cpu的矩阵输出结果
│   ├── C_gpu_100.bin: 规模为100*100的cuda的矩阵输出结果
│   ├── C_gpu_1000.bin: 规模为1000*1000的cuda的矩阵输出结果
│   ├── C_gpu_1500.bin: 规模为1500*1500的cuda的矩阵输出结果
│   ├── C_gpu_200.bin: 规模为200*200的cuda的矩阵输出结果
│   ├── C_gpu_2000.bin: 规模为2000*2000的cuda的矩阵输出结果
│   ├── C_gpu_3000.bin: 规模为3000*3000的cuda的矩阵输出结果
│   ├── C_gpu_500.bin: 规模为500*500的cuda的矩阵输出结果
│   ├── C_gpu_5000.bin: 规模为5000*5000的cuda的矩阵输出结果
│   ├── gpuMatrix: cuda的矩阵乘法可执行文件
│   ├── matrix_cpu: 串行cpu的矩阵乘法可执行文件
│   ├── matrix_cpu.cpp: 串行cpu的矩阵乘法源文件
│   ├── matrix_gpu.cu: cuda的矩阵乘法源文件
│   ├── runGpu.sh: 运行cuda的矩阵乘法脚本
│   ├── times.txt: cpu时间记录文件
│   ├── times_gpu.txt: cuda时间记录文件
│   └── times_gpu_include_memcpy_time.txt: cuda时间记录文件（包括cudamemcpy时间）
└── mpich: mpich所在文件夹
    ├── compile.sh: 编译mpich的矩阵乘法脚本
    ├── C_mpi_size_100.txt: 规模为100*100的mpich的矩阵输出结果
    ├── C_mpi_size_1000.txt: 规模为1000*1000的mpich的矩阵输出结果
    ├── C_mpi_size_1500.txt: 规模为1500*1500的mpich的矩阵输出结果
    ├── C_mpi_size_200.txt: 规模为200*200的mpich的矩阵输出结果
    ├── C_mpi_size_2000.txt: 规模为2000*2000的mpich的矩阵输出结果
    ├── C_mpi_size_3000.txt: 规模为3000*3000的mpich的矩阵输出结果
    ├── C_mpi_size_500.txt: 规模为500*500的mpich的矩阵输出结果
    ├── C_mpi_size_5000.txt: 规模为5000*5000的mpich的矩阵输出结果
    ├── C_mpi_time.txt: mpich时间记录文件
    ├── mpich_matrix: mpich的矩阵乘法可执行文件
    ├── mpich_matrix.cpp: mpich的矩阵乘法源文件
    └── run.sh: 运行mpich的矩阵乘法脚本


RGB: 
├── chuan.cpp: 基于串行cpu的rgb转灰度图
├── cpl_chuan.sh: 编译串行cpu的rgb转灰度图的脚本
├── chuanRgb: 串行cpu的rgb转灰度图的可执行文件
├── rgb_gpu.cu: 基于cuda的rgb转灰度图
├── cpl_gpu.sh: 编译cuda的rgb转灰度图的脚本
├── gpuRgb: cuda的rgb转灰度图的可执行文件
├── runGpu.sh: 运行cuda的rgb转灰度图的脚本
├── rgb_cpu.cpp: 基于opencv的cpu的rgb转灰度图
├── cpl_cpu.sh: 编译opencv的cpu的rgb转灰度图的脚本
├── cpuRgb: opencv的cpu的rgb转灰度图的可执行文件
├── test.jpg: 测试图片
├── gpu_output.jpg: cuda的rgb转灰度图的输出图片
├── test.jpg_cpu_output_copy.jpg: opencv的cpu的rgb转灰度图的输出图片
├── test.jpg_cpu_output.jpg: 串行的cpu的rgb转灰度图的输出图片
└── RGBtime.txt: 输出时间记录文件

project_KNN
├── serial.cpp: 串行的KNN代码
├── serial.sh: 串行KNN代码编译运行shell程序
├── parallel.cu: 并行的KNN代码
├── parallel.sh: 并行KNN的编译运行shell程序
├── banana.txt: banana训练集
├── banana_test.txt: banana测试集
├── newthyroif.txt: 甲状腺训练集
├── thy_text.txt: 甲状腺测试集
└── ring.txt: ring训练集
└── ring_test.txt: ring测试集
