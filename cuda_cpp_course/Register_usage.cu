#include "Header.h"

__global__ void register_usage_test(int * result, int size)
{
	int gid = threadIdx.x + blockIdx.x * blockDim.x;

	int x1 = 3465;
	int x2 = 1768;
	int x3 = 453;
	int x4 = x1 + x2 + x3;

	if (gid < size)
	{
		result[gid] = x4;
	}
}

//int main()
//{
//	int size = 1 << 22;
//	int byte_size = sizeof(int);
//
//	int* href = (int*)malloc(byte_size);
//	int* d_results;
//	cudaMalloc((void**)&d_results, byte_size);
//	cudaMemset(d_results, 0, byte_size);
//
//	dim3 blocks(128);
//	dim3 grid((size + blocks.x - 1) / blocks.x);
//
//	printf("Launching the kernel \n");
//	register_usage_test << <grid, blocks >> > (d_results, size);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(href, d_results, byte_size, cudaMemcpyDeviceToHost);
//
//	printf("Finished \n");
//	return 0;
//}