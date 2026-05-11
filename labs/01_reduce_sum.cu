#include "grpo/cuda_check.cuh"

#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

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

__global__ void sum_block_reduce_kernel(const float* x, float* partial, int n){
    // FAST SHARED MEMORY
    // extern as we don't know the size at compile time we just tell
    // the gpu how much memory to allocatr when we launch the kernel from the host
    // ultra-fast, only visible to threads within this block
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int base = blockIdx.x*blockDim.x*2;
    int i = base + tid;

    float v = 0.0f;
    // every thread grabs it's first element
    if(i<n){
        v+=x[i];
    }
    // and reaches exactly one block width forward to grab the 2nd ele
    if(i+blockDim.x<n){
        v+=x[i+blockDim.x];
    }

    sh[tid]=v;

    __syncthreads();

    // we start with half the block size
    // after every round, we divide stride by 2 
    for(int stride=blockDim.x/2; stride>0; stride>>=1){
        // Only threads "in the first half" of the current bracket do work
        // The others sit this round out
        if(tid<stride){
            sh[tid]+=sh[tid + stride];
        }
      __syncthreads();
    }

    //The total sum for this ENTIRE block is now sitting in sh[0]
    if (tid==0){
        partial[blockIdx.x]=sh[0];
    }
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

float sum_cuda_block_reduce(const float* d_x, int n, float* elapsed_ms){
    int threads = 256;
    // each thread has double duties, so each block
    // handles thread*2 elements. We need to round up
    int max_blocks = (n + (threads*2) - 1)/(threads*2);

    // A single kernel launch only reduces the array by a factor of threads*2
    // For ~4M numbers, we will end up with ~8000 partial sums
    // we need to keep feeding the partials back into the kernel until we have 1 number
    float* d_buff_a=nullptr;
    float* d_buff_b=nullptr; 
    CUDA_CHECK(cudaMalloc((void**)&d_buff_a, max_blocks*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_buff_b, max_blocks*sizeof(float)));

    // in looks at the original array, out points at buff a
    const float* in=d_x;
    float* out = d_buff_a;
    int curr_n=n;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // reduction loop
    while(curr_n>1){
        // calculate the number of blocks needed for this step
        int blocks = (curr_n + (threads*2) - 1)/(threads*2);

        // KERNEL LAUNCH
        // Note: the 3rd argument is for dynamic shared memory size
        // We are allocating enough memory for every thread in the block
        sum_block_reduce_kernel<<<blocks, threads, threads*sizeof(float)>>>(in, out, curr_n);
        CUDA_CHECK(cudaGetLastError());

        // next round
        curr_n = blocks;

        // what was out array this round becomes the in array for the next round
        in = out;

        // flip 'out' to whatever buffer we AREN'T currently using as input
        out = (out == d_buff_a) ? d_buff_b : d_buff_a; 
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));

    float res=0.0f;
    CUDA_CHECK(cudaMemcpy(&res, in, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_buff_a));
    CUDA_CHECK(cudaFree(d_buff_b));

    return res;
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

    float block_ms=0.0f;
    float block_sum=sum_cuda_block_reduce(d_x, n, &block_ms);

    auto rel_err = [](double got, double want){
        return std::fabs(got-want)/std::max(1.0, std::fabs(want));
    };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "n = " << n << "\n\n";

    std::cout << "CPU reference:\n";
    std::cout << "  sum = " << ref << "\n";
    std::cout << "  time ms = " << cpu_ms << "\n\n";

    std::cout << "CUDA atomic reduction:\n";
    std::cout << "  sum = " << atomic_sum << "\n";
    std::cout << "  rel err = " << rel_err(atomic_sum, ref) << "\n";
    std::cout << "  time ms = " << atomic_ms << "\n\n";

    std::cout << "CUDA block reduction:\n";
    std::cout << "  sum = " << block_sum << "\n";
    std::cout << "  rel err = " << rel_err(block_sum, ref) << "\n";
    std::cout << "  time ms = " << block_ms << "\n";

    CUDA_CHECK(cudaFree(d_x));
    return 0;
}
