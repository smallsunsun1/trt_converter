#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <algorithm>

#include "trt_converter/common/device.h"

using namespace sss;

__global__ void Compute(int *a, uint32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", a[i]);
} 

int main(int argc, char* argv[]) {
    cudaSetDevice(0);
    TRTCudaGraph graph;
    TRTCudaStream stream;
    TRTCudaEvent event;
    uint32_t size = std::stoi(argv[1]);
    int* data = new int[size];
    std::fill(data, data + size, 100);
    int* devData;
    CudaStatusCheck(cudaMalloc(&devData, size * sizeof(int)));
    graph.BeginCapture(stream);
    CudaStatusCheck(cudaMemcpyAsync((void*)devData, data, sizeof(int) * size, cudaMemcpyHostToDevice, stream.Get()));
    void* args[] = {&devData, &size};
    CudaStatusCheck(cudaLaunchKernel((void*)Compute, dim3(1), dim3(size), args, 0, stream.Get()));
    graph.EndCapture(stream);
    graph.Launch(stream);
    stream.Synchronize();
    CudaStatusCheck(cudaFree(devData));
    delete []data;
}
