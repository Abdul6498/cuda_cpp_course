#include "Header.h"

__global__ void code_without_divergence()
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	float a, b;
	a = b = 0;

	int warp_id = gid / 32;

	if (warp_id % 2 == 0)
	{
		a = 100.0;
		b = 50.0;
	}
	else
	{
		a = 200;
		b = 75;
	}

}

__global__ void divergence_code()
{
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

	float a, b;
	a = b = 0;

	if (gid % 2 == 0)
	{
		a = 100.0;
		b = 50.0;
	}
	else
	{
		a = 200;
		b = 75;
	}

}
//
//int main() {
//
//	printf("\n---------------------- WARP DIVERGENCE EXAMPLE ---------------------- \n\n");
//
//	size_t size = 1 << 22; //2^22
//
//	dim3 block_size(128);
//	dim3 grid_size((size + block_size.x - 1) / block_size.x);
//
//	code_without_divergence << <grid_size, block_size >> > ();
//	cudaDeviceSynchronize();
//
//	divergence_code << <grid_size, block_size >> > ();
//	cudaDeviceSynchronize();
//
//	cudaDeviceReset();
//	printf("Finished");
//
//	return 0;
//}