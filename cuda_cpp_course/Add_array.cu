#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <chrono>

__global__ void add(int* a, int *b, int *c, int size) {

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}
	//printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x :%d, gid: %d, c: %d\n", blockIdx.x , blockDim.x , threadIdx.x,  gid, c[gid]);

}

void add_cpu(int* a, int* b, int* c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
		//printf("i: %d, c: %d \n", i, c[i]);
	}
	// printf("i: %d, c: %d \n", i, c[i]);
}

//void compare_arrays(int* a, int* b, int size)
//{
//	for (int i = 0; i < size; i++)
//	{
//		//printf("i: %d, a: %d, b: %d\n", i, a[i], b[i]);
//		if (a[i] != b[i])
//		{
//			//printf("i: %d, a: %d, b: %d\n", i, a[i], b[i]);
//			printf("Arrays are different \n");
//			return;
//		}
//	}
//	printf("Arrays are same \n");
//}

//int main()
//{
//	int array_size = 1000000;
//
//	int block_size = 1024;
//
//	cudaError error;	//intialize cuda error function,
//	int size_in_bytes = array_size * sizeof(int);
//
//	//host pointers
//	int* h_a, * h_b, * gpu_results, *cpu_results;
//
//	//memory allocation in host
//	h_a = (int*)malloc(size_in_bytes);	
//	h_b = (int*)malloc(size_in_bytes); 
//	gpu_results = (int*)malloc(size_in_bytes);
//	cpu_results = (int*)malloc(size_in_bytes);
//
//	//intialize host pointer
//	time_t t;
//	srand((unsigned)time(&t));
//
//	for (size_t i = 0; i < array_size; i++)
//	{
//		h_a[i] = (int)(rand() & 0xFF);
//	}
//	for (size_t i = 0; i < array_size; i++)
//	{
//		h_b[i] = (int)(rand() & 0xFF);
//	}
//
//	//remove garbadge value and intialize to 0
//	memset(gpu_results, 0, size_in_bytes);
//	memset(cpu_results, 0, size_in_bytes);
//
//	//Cpu function call, sum function
//	//clock_t cpu_start, cpu_end; //two clock variables to check execution time on cpu
//
//	auto cpu_start = std::chrono::high_resolution_clock::now();
//	add_cpu(h_a, h_b, cpu_results, array_size);
//	auto cpu_end = std::chrono::high_resolution_clock::now();
//
//	//device pointer
//	int* d_a, * d_b, * d_c;
//	error = cudaMalloc((int**)&d_a, size_in_bytes); //get return error from cuda. Use cuda error function
//	if (error != cudaSuccess)
//	{
//		fprintf(stderr, " Error : %s \n", cudaGetErrorString); //cuda get error from function
//	}
//	error = cudaMalloc((int**)&d_b, size_in_bytes);
//	if (error != cudaSuccess)
//	{
//		fprintf(stderr, " Error : %s \n", cudaGetErrorString); //cuda get error from function
//	}
//	error = cudaMalloc((int**)&d_c, size_in_bytes);
//	if (error != cudaSuccess)
//	{
//		fprintf(stderr, " Error : %s \n", cudaGetErrorString); //cuda get error from function
//	}
//
//	//Copy data
//	//clock_t htod_start, htod_end;
//	auto htod_start = std::chrono::high_resolution_clock::now();
//	cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, h_b, size_in_bytes, cudaMemcpyHostToDevice);
//	auto htod_end = std::chrono::high_resolution_clock::now();
//
//	//launching the grid
//	dim3 block(block_size);
//	dim3 grid((array_size / block.x) +1);
//
//	//clock_t gpu_start, gpu_end;
//	auto gpu_start = std::chrono::high_resolution_clock::now();
//	add << <grid, block >> > (d_a, d_b, d_c, array_size);
//	auto gpu_end = std::chrono::high_resolution_clock::now();
//
//	cudaDeviceSynchronize();
//
//	//copy results back to host
//	//clock_t dtoh_start, dtoh_end;
//	auto dtoh_start = std::chrono::high_resolution_clock::now();
//	cudaMemcpy(gpu_results, d_c, size_in_bytes, cudaMemcpyDeviceToHost);
//	auto dtoh_end = std::chrono::high_resolution_clock::now();
//
//
//	printf("Sum function execution time on CPU: %d micro sec \n", std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
//	printf("Sum function execution time on GPU: %d micro sec \n", std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count());
//	printf("Memory transfer from host to device, Upload time: %d micro sec \n", std::chrono::duration_cast<std::chrono::microseconds>(htod_end - htod_start).count());
//	printf("Memory transfer from device to host, Download time: %d micro sec \n", std::chrono::duration_cast<std::chrono::microseconds>(dtoh_end - dtoh_start).count());
//
//	//results comparison
//	compare_arrays(cpu_results, gpu_results, array_size);
//
//	//free cuda occupied memory
//	cudaFree(d_c);
//	cudaFree(d_b);
//	cudaFree(d_a);
//
//	//free host memory
//	free(gpu_results);
//	free(h_a);
//	free(h_b);
//
//	cudaDeviceReset();
//
//	std::cout << "Finished" << std::endl;
//	return 0;
//}