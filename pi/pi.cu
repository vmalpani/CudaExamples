/*
  Accelereyes

  Monte Carlo Pi Estimation

  Estimate pi by calculating the ratio of points that fell inside of a
  unit circle with the points that did not.
*/

#include <iostream>
#include <vector>
#include "../common.h"

using namespace std;

static const int nsamples = 1e8;
static const int nthreads = 500;
static const int nitemsperthread = 1000;
static const int nblocks = nsamples / (nthreads * 1000);


// Create a kernel to estimate pi
__global__ void pi_optimized(float* x, float* y, int* global_count) {
    __shared__ int counts[nthreads];

    //int globalId = blockIdx.x * blockDim.x + nitemsperthread * threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    int thread_count=0;
    for (int i=0; i<nitemsperthread; i++) {
        int idx = globalId+(i*nthreads*nblocks);
        if (idx < nsamples) {
            if (x[idx]*x[idx] + y[idx]*y[idx] < 1.0) {
                thread_count++;
            }
        }
    }

    counts[threadIdx.x] = thread_count; 
    __syncthreads();

    if (threadIdx.x == 0) {
        int block_count = 0;
        for (int i=0; i<nthreads; i++) {
            block_count += counts[i];
        }
        global_count[blockIdx.x] = block_count;
    }
}


// Create a kernel to estimate pi
__global__ void pi_random(float* x, float* y, int* global_count) {
    __shared__ int counts[nthreads];

    int globalId = blockIdx.x * blockDim.x + nitemsperthread * threadIdx.x;

    int thread_count=0;
    for (int i=0; i<nitemsperthread; i++) {
        if (globalId+i < nsamples) {
            if (x[globalId+i]*x[globalId+i] + y[globalId+i]*y[globalId+i] < 1.0) {
                thread_count++;
            }
        }
    }

    counts[threadIdx.x] = thread_count; 
    __syncthreads();

    if (threadIdx.x == 0) {
        int block_count = 0;
        for (int i=0; i<nthreads; i++) {
            block_count += counts[i];
        }
        global_count[blockIdx.x] = block_count;
    }
}

int main(void)
{
    // allocate space to hold random values
    vector<float> h_randNumsX(nsamples);
    vector<float> h_randNumsY(nsamples);

    srand(time(NULL)); // seed with system clock
    
    //Initialize vector with random values
    for (int i = 0; i < h_randNumsX.size(); ++i) {
        h_randNumsX[i] = float(rand()) / RAND_MAX;
        h_randNumsY[i] = float(rand()) / RAND_MAX;
    }

    // Send random values to the GPU
    size_t size = nsamples * sizeof(float);
    float* d_randNumsX;
    float* d_randNumsY;
    int* global_counter;
    cudaMalloc(&d_randNumsX, size);  // TODO check return cuda* return codes
    cudaMalloc(&d_randNumsY, size);
    cudaMalloc(&global_counter, nblocks*sizeof(int));  // TODO check return cuda* return codes

    cudaMemcpy(d_randNumsX, &h_randNumsX.front(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_randNumsY, &h_randNumsY.front(), size, cudaMemcpyHostToDevice);


    // TODO launch kernel to count samples that fell inside unit circle
    int* result = (int*)malloc(nblocks*sizeof(int));;
    int nsamples_in_circle = 0;

    cudaEvent_t start, stop;
    float time;

    // Time pi consecutive access
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    pi_optimized <<< nblocks, nthreads >>> (d_randNumsX, d_randNumsY, global_counter);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // get the results from the GPU
    cudaMemcpy(result, global_counter, nblocks*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i < nblocks; i++) {
        nsamples_in_circle += result[i];
    }

    // fraction that fell within (quarter) of unit circle
    float estimatedValue = 4.0 * float(nsamples_in_circle) / nsamples;

    cout << "Optimized Pi" << endl;
    cout << "Estimated Value: " << estimatedValue << endl;
    cout << "Estimated Time: " << time << endl;

    cudaFree(d_randNumsX);
    cudaFree(d_randNumsY);
    cudaFree(global_counter);
}
