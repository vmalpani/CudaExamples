#include <iostream>

// parameter describing the size of matrix A
const int rows = 4096;
const int cols = 4096;

const int BLOCK_SIZE = 16;

// naive transpose kernel
__global__ void matrixTransposeNaive(float *_a,   // pointer to matrix A on the device
                                     float *_b)   // pointer to matrix B on the device
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col

    int index_in = i*cols+j;   // (i,j) from matrix A
    int index_out = j*rows+i;  // becomes (j,i) in matrix B = transpose(A)

    _b[index_out] = _a[index_in];
}

// coalesced memory transpose kernel
__global__ void matrixTransposeShared(float *_a,   // pointer to matrix A on the device
                                      float *_b)   // pointer to matrix B on the device
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int index_in = i*cols+j;   // (i,j) from matrix A

    // this thread fills in the appropriate box inside the shared memory in this block
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    tile [ threadIdx.y ] [ threadIdx.x ] = _a [index_in];

    // wait until all threads in this block are done writing to shared memory in parallel
    __syncthreads();

    int index_out = i*rows+j;  // (i,j) from matrix A becomes (j,i) in matrix B = transpose(A)

    _b[index_out] = tile[ threadIdx.x ] [ threadIdx.y ]; 
}

// coalesced memory transpose kernel without banking conflicts
__global__ void matrixTransposeNoBankConflicts(float *_a,   // pointer to matrix A on the device
                                      float *_b)   // pointer to matrix B on the device
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int index_in = i*cols+j;   // (i,j) from matrix A

    // this thread fills in the appropriate box inside the shared memory in this block
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];
    tile [ threadIdx.y ] [ threadIdx.x ] = _a [index_in];

    // wait until all threads in this block are done writing to shared memory in parallel
    __syncthreads();

    int index_out = i*rows+j;  // (i,j) from matrix A becomes (j,i) in matrix B = transpose(A)

    _b[index_out] = tile[ threadIdx.x ] [ threadIdx.y ]; 
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
            a[row + col*rows] = 2.0; //row + col*rows;
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

//  matrixTransposeNaive<<<dimGrid,dimBlock>>>(_a, _b);
//  matrixTransposeShared<<<dimGrid,dimBlock>>>(_a, _b);
//  matrixTransposeNoBankConflicts<<<dimGrid,dimBlock>>>(_a, _b);

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

    // free host memory
    delete a;
    delete b;

    // successful program termination
    return 0;
}
