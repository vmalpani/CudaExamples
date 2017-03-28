#include <iostream>
#include <assert.h>
#include <cstdio>

using namespace std;

// parameter describing the size of matrix A
const int rows = 4096;
const int cols = 4096;

const int BLOCK_SIZE = 16;

// transpose shared kernel
__global__ void transpose_shared(float* a, float*b) {
  __shared__ float result[BLOCK_SIZE][BLOCK_SIZE+1];
  // adding one to avoid bank conflict

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int width = gridDim.x * blockDim.x;
  int height = gridDim.y * blockDim.y;

   // perform transpose
   if (x < height && y < width) {
       result[threadIdx.x][threadIdx.y] = a[y*height + x];
   }
    __syncthreads();
   b[x*height + y] = result[threadIdx.x][threadIdx.y];  
} 

// transpose kernel
__global__ void transpose_naive(float* a, float*b) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int width = gridDim.x * blockDim.x;
  int height = gridDim.y * blockDim.y;

   // perform transpose
   if (x < height && y < width) {
       b[x*height + y] = a[y*width + x];
   }
} 

void print_matrix(float* mat, int rows, int cols) {
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            printf("%2.2f ", mat[row*cols + col]);
        }
        printf("\n");
    }
}

// the main program starts life on the CPU and calls device kernels as required
int main(int argc, char *argv[])
{
    // allocate space in the host for storing input arrays (a and b) and the output array (c)
    float *a = new float[rows*cols];
    float *b = new float[rows*cols];

    // define device pointers for the same arrays when they'll be copied to the device
    float *_a, *_b;

    // allocate memory on the device (GPU) and check for errors (if any) during this call
    cudaError_t err;

    // allocate space for matrix A 
    err = cudaMalloc((void **) &_a, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    // allocate space for matrix B
    err = cudaMalloc((void **) &_b, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    // Fill matrix A
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            a[row + col*rows] = row + col*rows;
        }
    }


    // Copy array contents of A from the host (CPU) to the device (GPU)
    // Note that this is copied to the "global" memory on the device and is accessible to all threads in all blocks
    cudaMemcpy(_a, a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    // assign a 2D distribution of 16 x 16 x 1 CUDA "threads" within each CUDA "block"
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid( ceil(float(rows)/float(dimBlock.x)), ceil(float(cols)/float(dimBlock.y)), 1 );

    float time;

    // create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord( start, 0);

    // launch the GPU kernel
    // transpose_naive <<< dimGrid, dimBlock >>> (_a, _b);
    transpose_shared <<< dimGrid, dimBlock >>> (_a, _b);

    // stop the timer
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop);

    // print out the time required for the kernel to finish the transpose operation
    double Bandwidth = 2.0*1000*(double)(rows*cols*sizeof(float)) / (1000*1000*1000*time);
    std::cout << "Elapsed Time  = " << time << " Bandwidth used (GB/s) = " << Bandwidth << std::endl;

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(b, _b, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(_a);
    cudaFree(_b);

    // print results
    print_matrix(a, rows, cols);
    printf("\n");
    print_matrix(b, rows, cols);

    // free host memory
    delete a;
    delete b;


    // successful program termination
    return 0;
}
