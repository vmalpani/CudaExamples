/*************************************************
** Accelereyes Training Day 1					**
** Vector Addition								**
**						 						**
** This program will add two vectors and store  **
** the result in a third vector using the GPU	**
*************************************************/

#include <iostream>
#include <vector>
#include "cuda.h"

__global__ void add(int* a, int* b, int* c) {
    // calculate global id
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // perform calculation
    c[id] = a[id] + b[id];
}

int main(void) {
    using namespace std;
    long N = 10000;
    int blocks = 10;
    size_t size = N * sizeof(int);

    // initalize data on hos
    vector<int> h_a(N, 1);
    vector<int> h_b(N, 2);
    vector<int> h_c(N);

    // initialize device pointers and allocate memory on the GPU
    int* device_pointer_a;
    cudaMalloc(&device_pointer_a, size);

    int* device_pointer_b;
    cudaMalloc(&device_pointer_b, size);

    int* device_pointer_c;
    cudaMalloc(&device_pointer_c, size);

    // move host data to the GPU
    cudaMemcpy(device_pointer_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_pointer_b, h_b.data(), size, cudaMemcpyHostToDevice);

    // launch kernel
    add <<< blocks, 1000 >>> (device_pointer_a, device_pointer_b, device_pointer_c);

    // get the results from the GPU
    cudaMemcpy(h_c.data(), device_pointer_c, size, cudaMemcpyDeviceToHost);

    cudaFree(device_pointer_a);
    cudaFree(device_pointer_b);
    cudaFree(device_pointer_c);

    // print results
	for(int i = 0; i < N; ++i) {
		cout << h_c[i] << ", ";
	}

	return 0;
}
