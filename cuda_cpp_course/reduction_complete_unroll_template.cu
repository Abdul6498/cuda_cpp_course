#include "Header.h"

template<unsigned int iblock_size>
__global__ void reduction_kernel_complete_unrolling_template(int* int_array, int* temp_array, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int* i_data = int_array + blockDim.x * blockIdx.x;
	if (gid > size)
		return;

	if (iblock_size == 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];
	__syncthreads();

	if (iblock_size == 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];
	__syncthreads();

	if (iblock_size == 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];
	__syncthreads();

	if (iblock_size == 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();

	if (iblock_size == 64 && tid < 32)
		i_data[tid] += i_data[tid + 32];
	__syncthreads();

	if (tid < 32)
	{
		volatile int* vsmem = i_data; //memory load and store directly in the global memory without any cashes
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
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
//	int size = 1 << 27; //128 Mb of data
//	int byte_size = size * sizeof(int);
//	int block_size = 128;
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
//	switch (block_size)
//	{
//	case 1024:
//		reduction_kernel_complete_unrolling_template <1024> << <grid, block >> > (d_input, d_temp, size);
//		break;
//	case 512:
//		reduction_kernel_complete_unrolling_template <512> << <grid, block >> > (d_input, d_temp, size);
//		break;
//	case 256:
//		reduction_kernel_complete_unrolling_template <256> << <grid, block >> > (d_input, d_temp, size);
//		break;
//	case 128:
//		reduction_kernel_complete_unrolling_template <128> << <grid, block >> > (d_input, d_temp, size);
//		break;
//	case 64:
//		reduction_kernel_complete_unrolling_template <64> << <grid, block >> > (d_input, d_temp, size);
//		break;
//	}
//	
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