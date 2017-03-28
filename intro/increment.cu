// simple increment kernel

#include <cuda.h>
#include <stdio.h>

//TODO: increment kernel
__global__
void increment(float *val) {
    *val += 2;
}

int main(void)
{
    // create host array and initialize
    float* device_pointer;
    
    // allocate device memory    
    cudaMalloc(&device_pointer, sizeof(float));
    
    // print original value
    float* host_pointer = (float*)malloc(sizeof(float));
    *host_pointer = 40;
    printf("%f\n", *host_pointer);

    // memcpy to device
    cudaMemcpy(device_pointer, host_pointer, sizeof(float), cudaMemcpyHostToDevice);

    // launch the increment kernel
    increment <<< 1, 1 >>> (device_pointer);

    // memcpy results back to host
    cudaMemcpy(host_pointer, device_pointer, sizeof(float), cudaMemcpyDeviceToHost);

    // print new value
    printf("%f\n", *host_pointer);

    cudaFree(device_pointer);

    return 0;
}
