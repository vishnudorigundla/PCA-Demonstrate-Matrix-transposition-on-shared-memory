# PCA-Demonstrate-Matrix-transposition-on-shared-memory
Comparing the Performance of the Rectangular Shared Memory Kernels with  grid (1,1) block (16,16)
## Aim:
To demonstrate the Matrix transposition on shared memory with grid (1,1) block (16,16)
## Procedure:

Step 1 : Include the required files and library.

Step 2 : Define the block size to be 16 .

Step 3 : Intoduce a void function to print the data.

Step 4 : Introduce global functions to set and read the row & column. In the function , decalre a shared memory , map thr thread index to global memory index , perform store operation and wait for all threads to complete and them perform load operation.

Step 5 : Inroduce the main function, in the main method set up the device ,array size and declare the execution configuration. Allocate the device memory and finally free the host and device memory followed by reseting the device.

Step 6 : Save and execute the program.

### Program:
```
Developed By: G venkata Pavan Kumar
Reg. No.: 212221240013
```
```
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using shared memory to transpose rectangular thread coordinates
 * of a CUDA grid into a global memory array. Different kernels below
 * demonstrate performing reads and writes with different ordering, as well as
 * optimizing using memory padding.
 */

#define BDIMX 16
#define BDIMY 16
#define IPAD  2

void printData(char *msg, int *in,  const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%4d", in[i]);
        fflush(stdout);
    }

    printf("\n\n");
}

__global__ void setRowReadRow(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x] ;
}

__global__ void setColReadCol(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadCol2(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[icol][irow] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow] ;
}

__global__ void setRowReadCol(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColPad(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow] ;
}

__global__ void setRowReadColDyn(int *out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * blockDim.x + irow;

    // shared memory store operation
    tile[idx] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[col_idx];
}

__global__ void setRowReadColDynPad(int *out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    // shared memory store operation
    tile[row_idx] = g_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig ( &pConfig ));
    printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

    // set up array size
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argc > 1) iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    dim3 grid  (1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
            block.y);

    // allocate device memory
    int *d_C;
    CHECK(cudaMalloc((int**)&d_C, nBytes));
    int *gpuRef  = (int *)malloc(nBytes);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadRow       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setColReadCol       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol2<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setColReadCol2      ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadCol       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMX*BDIMY*sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadColDyn    ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadColPad    ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + IPAD)*BDIMY*sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadColDynPad ", gpuRef, nx * ny);

    // free host and device memory
    CHECK(cudaFree(d_C));
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
```
## Output:
```
root@MidPC:/home/student/Desktop# nvcc test.cu
root@MidPC:/home/student/Desktop# ./a.out
./a.out at device 0: NVIDIA GeForce GTX 1660 SUPER with Bank Mode:4-Byte <<< grid (1,1) block (16,16)>>>
root@MidPC:/home/student/Desktop# nvprof ./a.out
==14603== NVPROF is profiling process 14603, command: ./a.out
./a.out at device 0: NVIDIA GeForce GTX 1660 SUPER with Bank Mode:4-Byte <<< grid (1,1) block (16,16)>>>
==14603== Profiling application: ./a.out
==14603== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   35.10%  7.8410us         7  1.1200us  1.1200us  1.1210us  [CUDA memcpy DtoH]
                   32.52%  7.2640us         7  1.0370us     960ns  1.4720us  [CUDA memset]
                    4.73%  1.0560us         1  1.0560us  1.0560us  1.0560us  setRowReadCol(int*)
                    4.73%  1.0560us         1  1.0560us  1.0560us  1.0560us  setColReadCol2(int*)
                    4.73%  1.0560us         1  1.0560us  1.0560us  1.0560us  setRowReadColDyn(int*)
                    4.58%  1.0240us         1  1.0240us  1.0240us  1.0240us  setRowReadColDynPad(int*)
                    4.58%  1.0240us         1  1.0240us  1.0240us  1.0240us  setColReadCol(int*)
                    4.58%  1.0240us         1  1.0240us  1.0240us  1.0240us  setRowReadColPad(int*)
                    4.44%     992ns         1     992ns     992ns     992ns  setRowReadRow(int*)
      API calls:   74.98%  139.00ms         1  139.00ms  139.00ms  139.00ms  cudaDeviceGetSharedMemConfig
                   24.49%  45.404ms         1  45.404ms  45.404ms  45.404ms  cudaDeviceReset
                    0.12%  215.90us        97  2.2250us     170ns  93.829us  cuDeviceGetAttribute
                    0.11%  195.87us         1  195.87us  195.87us  195.87us  cuDeviceTotalMem
                    0.10%  186.03us         1  186.03us  186.03us  186.03us  cudaGetDeviceProperties
                    0.06%  110.17us         1  110.17us  110.17us  110.17us  cudaMalloc
                    0.04%  78.960us         1  78.960us  78.960us  78.960us  cudaFree
                    0.04%  73.669us         7  10.524us  9.3100us  14.940us  cudaMemcpy
                    0.03%  50.060us         7  7.1510us  4.2900us  23.140us  cudaLaunchKernel
                    0.02%  33.690us         1  33.690us  33.690us  33.690us  cuDeviceGetName
                    0.02%  29.020us         7  4.1450us  2.5200us  12.920us  cudaMemset
                    0.00%  4.6900us         1  4.6900us  4.6900us  4.6900us  cuDeviceGetPCIBusId
                    0.00%  2.6800us         1  2.6800us  2.6800us  2.6800us  cudaSetDevice
                    0.00%  2.1600us         3     720ns     180ns  1.7500us  cuDeviceGetCount
                    0.00%     750ns         2     375ns     170ns     580ns  cuDeviceGet
                    0.00%     240ns         1     240ns     240ns     240ns  cuDeviceGetUuid
root@MidPC:/home/student/Desktop# 106
```
## Result:
The Matrix transposition on shared memory with grid (1,1) block (16,16) is demonstrated successfully.
