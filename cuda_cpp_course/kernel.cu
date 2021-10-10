
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//16x16 thread divided into 4 different threads of 8x8 threads. THis behaves like a nested loop with 4 inner loops

__global__ void print_details() {

    printf("threadIdx.x : %d, threadIdx.y : %d, threadidx.z : %d , blockDim.x : %d, blockDim.y : %d, gridDim.x : %d, gridDim.z : %d\n", 
        threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
}

//trasfer data from main function to gpu kernel function
__global__ void unique_idx_calc_threadIdx(int* input) {
	int tid = threadIdx.x;	//we need one variable and 2 dimension is required only
	printf("threadIdx : %d, value : %d \n", tid, input[tid]);
}

__global__ void unique_gid_calc_threadIdx(int* input)
{
	int tid = threadIdx.x;
	int gid = tid + (blockIdx.x * blockDim.x);
	printf("threadIdx.x: %d, blockIdx.x: %d , global Id : %d, value : %d \n",threadIdx.x, blockIdx.x, gid, input[gid]);
}
/*
int main()
{
	int array_size = 8; //size of array
	int array_byte_size = sizeof(int) * array_size; //size of array in bytes
	int h_data[] = { 23,9,4,53,65,12,1,33 };

	for (int i = 0; i < array_size; i++)
	{
		printf("%d ", h_data[i]);
	}
	printf("\n \n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(4);	//Numder of threads in x direction
	dim3 grid(2);	//Number of blocks

	unique_idx_calc_threadIdx << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	unique_gid_calc_threadIdx << <grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	/*
    int nx, ny;
    //total number of threads in x and y direction
    nx = 16;
    ny = 16;

    //size of each block in a bigger 2d array/block
    dim3 block(8, 8);
    //number of block in x direction and number of in y direction in this case its 2 and 2
    dim3 grid(nx / block.x, ny / block.y);

    
    print_details << < grid, block >> > (); //(number of blocks in x,y,z direction , size of each block)
	cudaDeviceSynchronize();
    */
/*    return 0;
}
*/