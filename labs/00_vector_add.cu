#include <algorithm>
#include <cmath> 
#include <iostream>
#include <vector>

#include "grpo/cuda_utils.cuh"

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n) c[i]=a[i]+b[i];
}

int main(){
    int n=1<<20;
    std::vector<float> a(n),b(n),c(n),ref(n);

    for(int i=0; i<n; i++){
        a[i]=0.001f*i;
        b[i]=0.001f*i;
        ref[i]=a[i]+b[i];
    }

    float* d_a=nullptr;
    float* d_b=nullptr;
    float* d_c=nullptr;

    CUDA_CHECK(cudaMalloc(&d_a,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c,n*sizeof(float)));

    // copying to gpu from cpu
    CUDA_CHECK(cudaMemcpy(d_a,a.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,b.data(),n*sizeof(float),cudaMemcpyHostToDevice));

    // workers 
    int threads = 256;
    int blocks = (n + threads - 1)/threads;

    // launch kernel
    vector_add_kernel<<<blocks,threads>>>(d_a,d_b,d_c,n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c.data(),d_c, n*sizeof(float),cudaMemcpyDeviceToHost));

    float max_err=0.0;
    for(int i=0;i<n;i++){
        max_err=std::max(max_err,std::abs(c[i]-ref[i]));
    }

    std::cout << "max error: " << max_err << "\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

}
