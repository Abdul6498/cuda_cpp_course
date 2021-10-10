#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

//unique index calculation for 2d grid
__global__ void unique_gid_calculation_2d(int* input)
{
	int tid = threadIdx.x;
	int block_offset = blockIdx.x * blockDim.x;
	int row_offset = gridDim.x * blockDim.x * blockIdx.y;
	int gid = tid + block_offset + row_offset;
	printf("blockDim.x: %d, gridDim.x: %d ,threadIdx.x: %d, blockIdx.x: %d ,block_offset: %d, row_offset: %d, global Id : %d, value : %d \n", 
		 blockDim.x,gridDim.x,threadIdx.x, blockIdx.x,block_offset, row_offset, gid, input[gid]);
}

//uinque index calculation of threads in 4 blocks and global ids
__global__ void unique_gid_calculation_2d_2d(int* input)
{
	int tid = blockDim.x * threadIdx.y + threadIdx.x;

	int num_threads_in_a_block = blockDim.x * blockDim.y;
	int block_offset = blockIdx.x * num_threads_in_a_block;

	int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
	int row_offset = num_threads_in_a_row * blockIdx.y;
	int gid = tid + block_offset + row_offset;
	printf("blockDim.x: %d, gridDim.x: %d ,threadIdx.x: %d, blockIdx.x: %d ,block_offset: %d, row_offset: %d, global Id : %d, value : %d \n",
		blockDim.x, gridDim.x, threadIdx.x, blockIdx.x, block_offset, row_offset, gid, input[gid]);
}

int main()
{
	int array_size = 16; //size of array
	int array_byte_size = sizeof(int) * array_size; //size of array in bytes
	int h_data[] = { 23,9,4,53,65,12,1,33, 14, 78,99,45,67,34,22, 11 };

	for (int i = 0; i < array_size; i++)
	{
		printf("%d ", h_data[i]);
	}
	printf("\n \n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(4);	//Numder of threads in x direction
	dim3 grid(2,2);	//Number of blocks in each direction

	//unique_gid_calculation_2d << <grid, block >> > (d_data);
	//cudaDeviceSynchronize();

	dim3 block_2d_2d(4);	//Numder of threads in x direction
	dim3 grid_2d_2d(2, 2);	//Number of blocks in each direction
	unique_gid_calculation_2d_2d << <grid_2d_2d, block_2d_2d >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();


	printf("Finished");
	return 0;
}