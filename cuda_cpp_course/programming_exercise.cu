#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_details_3d() {

	printf("blockDim.x : %d, blockDim.y : %d, blockDim.z : %d\n", blockDim.x, blockDim.y, blockDim.z);

	printf("gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);

	printf("threadIdx.x : %d, threadIdx.y : %d, threadidx.z : %d \n",
		threadIdx.x, threadIdx.y, threadIdx.z);
	
	printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);

	

}
/*
int main()
{
	// code for print_details_3d kernel
	int nx, ny, nz;
	nx = 8; //8 threads in x direction 4 per block
	ny = 8; //8 threads in y direction 4 per block
	nz = 8; //8 threads in z direction 4 per block

	dim3 block(4,4,4); //size of block in x,y and z dimension
	dim3 grid(nx / block.x, ny / block.y, nz / block.z);

	print_details_3d << <grid, block>> > ();
	cudaDeviceSynchronize();

	return 0;
}
*/