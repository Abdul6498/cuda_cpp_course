#include "Header.h"

// 2 blocks unrolling
__global__ void reduction_unrolling_blocks2(int* input, int* temp, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
	int index = BLOCK_OFFSET + tid;

	int* i_data = input + BLOCK_OFFSET;
	
	if (gid > size)
		return;
	if ((index + blockDim.x) < size)
	{
		input[index] += input[index + blockDim.x];
	}

	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
	{
		if (tid < offset)
		{
			i_data[gid] += i_data[gid + offset];
		}

		__syncthreads();
	}

	//store the partial sum
	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[gid];
	}
}

//4 block unrolling
__global__ void reduction_unrolling_blocks4(int* input, int* temp, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int BLOCK_OFFSET = blockIdx.x * blockDim.x * 4;
	int index = BLOCK_OFFSET + tid;

	int* i_data = input + BLOCK_OFFSET;

	if (gid > size)
		return;
	if ((index + 3 * blockDim.x) < size)
	{
		int a1 = input[index];
		int a2 = input[index + blockDim.x];
		int a3 = input[index + 2 * blockDim.x];
		int a4 = input[index + 3 * blockDim.x];

		input[index] = a1 + a2 + a3 + a4;
	}

	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
	{
		if (tid < offset)
		{
			i_data[gid] += i_data[gid + offset];
		}

		__syncthreads();
	}

	//store the partial sum
	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[gid];
	}
}

//int main()
//{
//	printf("Running neighbored pairs reduction kernel \n");
//
//	int size = 8; //128 Mb of data
//	int byte_size = size * sizeof(int);
//	int block_size = 4;
//	dim3 block(block_size);
//	dim3 grid((size / block.x)/2);	//for 2 block unrolling we divide by 2, for 4 block unrolling we divide by 4
//
//	int* h_input, * h_ref;
//	h_input = (int*)malloc(byte_size);
//
//	initialize(h_input, size, INIT_RANDOM);
//	//for (int i = 0; i < size; i++)
//	//{
//	//	printf("h_input[%d]: %d \n", i, h_input[i]);
//	//}
//	int cpu_result = reduction_cpu(h_input, size);
//
//
//
//	printf("Kernel launch parameter | grid.x : %d, block.x : %d \n", grid.x, block.x);
//
//	//partial sum array
//	int temp_array_byte_size = sizeof(int) * grid.x;
//	h_ref = (int*)malloc(temp_array_byte_size);
//
//	int* d_input, * d_temp;
//
//	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));
//
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));
//
//	printf("Test 1 \n");
//	reduction_unrolling_blocks2 << <grid, block >> > (d_input, d_temp, size);
//	printf("Test 2 \n");
//	gpuErrchk(cudaDeviceSynchronize());
//	printf("Test 3\n");
//
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	int gpu_result = 0;
//
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	//validity check
//	compare_results(gpu_result, cpu_result);
//
//	gpuErrchk(cudaFree(d_temp));
//	gpuErrchk(cudaFree(d_input));
//
//	free(h_ref);
//	free(h_input);
//
//	gpuErrchk(cudaDeviceReset());
//
//	printf("Finished \n");
//	return 0;
//}