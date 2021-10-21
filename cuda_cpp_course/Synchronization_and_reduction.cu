#include "Header.h"

__global__ void redunction_neighbored_pairs(int * input, int * temp, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x / 2; offset *=2)
	{
		if (tid % (2 * offset) == 0)
		{
			input[gid] += input[gid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = input[gid];
	}
}

__global__ void reduction_neighbored_pairs_improved(int* int_array, int* temp_array, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	//local data block pointer
	int* i_data = int_array + blockDim.x * blockIdx.x; //after + symbol chage the location of storage inside the pointer

	printf("gid : %d, i_data: %d, int_array: %d, blockDim.x: %d, blockIdx.x: %d, idata_size: %d \n",
		gid, i_data[gid], int_array[gid], blockDim.x, blockIdx.x, sizeof(i_data)/sizeof(int));

	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
	{
		int index = 2 * offset * tid;
		if (index < blockDim.x)
		{
			i_data[index] += i_data[index + offset];
			//printf("[IN IF] gid: %d, offset: %d, tid: %d, blockDim.x: %d, index: %d, i_data[%d] : %d \n",
			//	gid , offset, tid, blockDim.x, index, index, i_data[index]);
		}

		__syncthreads();
		//printf("[IN FOR] gid: %d, offset: %d, tid: %d, blockDim.x: %d, index: %d, i_data[%d] : %d \n",
		//	gid, offset, tid, blockDim.x, index, index, i_data[index]);
	}

	if (tid == 0)
	{
		temp_array[blockIdx.x] = int_array[gid];
		//printf("temp_array[%d]: %d, int_array[%d]: %d \n", blockIdx.x, temp_array[blockIdx.x], gid, int_array[gid]);
	}
}

__global__ void reduction_interleaved_pairs(int* int_array, int* temp_array, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size)
		return;

	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
	{
		if (tid < offset)
		{
			int_array[gid] += int_array[gid + offset];
		}

		__syncthreads();
	}

	//store the partial sum
	if (tid == 0)
	{
		temp_array[blockIdx.x] = int_array[gid];
	}
}

//int main()
//{
//	printf("Running neighbored pairs reduction kernel \n");
//
//	int size = 8; //128 Mb of data
//	int byte_size = size * sizeof(int);
//	int block_size = 2;
//	dim3 block(block_size);
//	dim3 grid(size / block.x);
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
//	reduction_neighbored_pairs_improved << <grid, block >> > (d_input, d_temp, size);
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