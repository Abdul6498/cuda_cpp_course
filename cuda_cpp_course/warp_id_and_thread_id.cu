#include "Header.h"

__global__ void print_details_of_warps()
{
	int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

	int warp_id = threadIdx.x / 32;

	int gbid = blockIdx.y * gridDim.x + blockIdx.x;	//global block index

	printf("tid : %d, bid.x : %d, bid.y : %d, gid : %d, wrap_id : %d, gbid : %d \n",
		threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

//int main()
//{
//	dim3 block_size(42);
//	dim3 grid_size(2, 2);
//
//	print_details_of_warps << <grid_size, block_size >> > ();
//	cudaDeviceSynchronize();
//
//	cudaDeviceReset();
//	printf("Finished \n");
//}