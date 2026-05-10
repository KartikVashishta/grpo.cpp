#include "grpo/cuda_check.cuh"

#include <vector>
#include <chrono>
#include <iostream>

double sum_cpu(const std::vector<float> &x){
    double s = 0.0;
    for(float v:x) s+=static_cast<double>(v);
    return s;
}

// fixed atomicAdd 
__global__ void sum_atomic_kernel(const float* x, float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n) atomicAdd(out, x[i]);
}

float sum_cuda_atomic(const float* d_x, int n, float* elapsed_ms){
    int threads = 256;
    int blocks = (n + threads - 1)/threads;

    float* d_out=nullptr;

    // allocating and setting memory for one float value
    CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(float))); 
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record, launch, record, sync and copy
    CUDA_CHECK(cudaEventRecord(start));
    sum_atomic_kernel<<<blocks, threads>>>(d_x, d_out, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // wait for the GPU to finish

    CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));

    float out=0.0f;

    //copy the single float back to the host
    CUDA_CHECK(cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    //cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));

    return out;
}
int main(){
    int n = 1<<22; //~4.2M
    std::vector<float> x(n);
    
    // An array with ~4.2M random numbers
    for(int i=0; i<n; i++) x[i]=1.0f/static_cast<float>(1+(i%100));

    // Timing accumulation on the CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    double ref = sum_cpu(x);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop-cpu_start).count();

    float* d_x = nullptr;
    //converting float** to void**
    CUDA_CHECK(cudaMalloc((void**)&d_x, n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    float atomic_ms=0.0f; 
    float atomic_sum=sum_cuda_atomic(d_x, n, &atomic_ms);

    std::cout << "CPU sum: " << ref << " (Time: " << cpu_ms << " ms)\n";
    std::cout << "Atomic sum: " << atomic_sum << " (Time: " << atomic_ms << " ms)\n";

    CUDA_CHECK(cudaFree(d_x));
    return 0;
}