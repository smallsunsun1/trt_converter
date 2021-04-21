#include <iostream>
#include "cuda_runtime_api.h"

int main(int argc, char* argv[]) {
    cudaSetDevice(0);
    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "elapsed_time: " << elapsed_time << std::endl;
}