#include "Header.h"

__global__ void test_sum_array_for_memory(float* a, float* b, float* c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
		c[gid] = a[gid] + b[gid];
}

//int main()
//{
//	printf("Running neighbored pairs reduction kernel \n");
//
//	int size = 1 << 27; //128 Mb of data
//	int byte_size = size * sizeof(int);
//	int block_size = 128;
//	dim3 block(block_size);
//	dim3 grid((size + block.x -1)/ block.x);	
//
//	int* h_input, * h_c, *h_b;
//	h_input = (int*)malloc(byte_size);
//	h_b = (int*)malloc(byte_size);
//
//	initialize(h_input, size, INIT_RANDOM);
//	initialize(h_b, size, INIT_RANDOM);
//	//for (int i = 0; i < size; i++)
//	//{
//	//	printf("h_input[%d]: %d \n", i, h_input[i]);
//	//}
//	//int cpu_result = reduction_cpu(h_input, size);
//
//
//
//	printf("Kernel launch parameter | grid.x : %d, block.x : %d \n", grid.x, block.x);
//
//	h_c = (int*)malloc(byte_size);
//	memset(h_c, 0, byte_size);
//
//	int* d_a, * d_b, *d_c;
//
//	gpuErrchk(cudaMalloc((void**)&d_a, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_b, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_c, byte_size));
//
//	gpuErrchk(cudaMemset(d_c, 0, byte_size));
//	
//	gpuErrchk(cudaMemcpy(d_a, h_input, byte_size, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
//
//	test_sum_array_for_memory << <grid, block >> > (d_a, d_b, d_c, size);
//
//	gpuErrchk(cudaDeviceSynchronize());
//
//	gpuErrchk(cudaMemcpy(h_c, d_c, byte_size, cudaMemcpyDeviceToHost));
//
//
//	//validity check
//	//compare_results(gpu_result, cpu_result);
//
//	gpuErrchk(cudaFree(d_a));
//	gpuErrchk(cudaFree(d_b));
//	gpuErrchk(cudaFree(d_c));
//
//	free(h_input);
//	free(h_b);
//	free(h_b);
//
//	gpuErrchk(cudaDeviceReset());
//
//	printf("Finished \n");
//	return 0;
//}