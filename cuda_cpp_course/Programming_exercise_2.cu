#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void mem_trs_3d(int * input, int size) {
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
	int tid_z = threadIdx.z;
	
	int tid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	int num_threads_in_a_block = blockDim.x * blockDim.y * blockDim.z;
	int block_offset = blockIdx.x * num_threads_in_a_block;
	int x_offset = num_threads_in_a_block * blockIdx.x;
	int y_offset = gridDim.x * num_threads_in_a_block * blockIdx.y;
	int z_offset = gridDim.y * gridDim.x * num_threads_in_a_block * blockIdx.z;
	int gid = tid + x_offset + y_offset + z_offset;
	if(gid < size)
		printf("BlockIdx.x: %d, BlockIdx.y: %d, blockidx.z: %d, tid: %d, number_of_threads_in_a_block: %d, block_offset: %d, gid: %d, value: %d \n",
			blockIdx.x, blockIdx.y, blockIdx.z , tid, num_threads_in_a_block, block_offset, gid, input[gid]);

}

int main() {

	int array_size = 64;
	int size_in_bytes = array_size * sizeof(int);
	int* h_input;
	int* d_input;

	//assign a memory in the host
	h_input = (int*)malloc(size_in_bytes);	//type casting

	//generating a random number
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < array_size; i++) {
			h_input[i] = (int)(rand() & 0xff);
	}

	cudaMalloc((void**)&d_input, size_in_bytes);

	cudaMemcpy(d_input, h_input, size_in_bytes, cudaMemcpyHostToDevice);
	int nx, ny, nz;
	nx = 4;
	ny = 4;
	nz = 4;

	dim3 block(2, 2, 2);
	dim3 grid(nx / block.x, ny / block.y, nz / block.z);

	mem_trs_3d << <grid, block >> > (d_input, array_size);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	printf("Finished");
	return 0;
}