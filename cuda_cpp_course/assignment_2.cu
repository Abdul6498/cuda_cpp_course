#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <chrono>
#include <cassert>

__global__ void add_3_gpu(int* a, int* b, int* c, int* result, int size)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	if (gid < size)
	{
		result[gid] = a[gid] + b[gid] + c[gid];
	}
}
void add_3_cpu(int* a, int* b, int* c, int* result, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = a[i] + b[i] + c[i];
	}
}
//
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

//int main(){
//	int array_size = 4194304;
//	int size_in_bytes = array_size * sizeof(int);
//	int block_size = 512;
//	printf("Enter block size: ");
//	block_size = std::cin.get();
//	assert(block_size > 0 && block_size % 2 == 0);
//	
//	//variables initlization
//	int* h_a, * h_b, * h_c, * cpu_result, * gpu_result;
//
//	//Memory allocation in cpu
//	h_a = (int*)malloc(size_in_bytes);
//	h_b = (int*)malloc(size_in_bytes);
//	h_c = (int*)malloc(size_in_bytes);
//	cpu_result = (int*)malloc(size_in_bytes);
//	gpu_result = (int*)malloc(size_in_bytes);
//
//	memset(gpu_result, 0, size_in_bytes);
//	memset(cpu_result, 0, size_in_bytes);
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
//	for (size_t i = 0; i < array_size; i++)
//	{
//		h_c[i] = (int)(rand() & 0xFF);
//	}
//
//	int * d_a, * d_b, * d_c, * gpu_result_k;
//
//	cudaError error;	//intialize cuda error function,
//
//	error = (cudaMalloc((int**)&d_a, size_in_bytes));
//	if (error != cudaSuccess)
//	{
//		fprintf(stderr, " Error : %s \n", cudaGetErrorString); //cuda get error from function
//	}
//	error = (cudaMalloc((int**)&d_b, size_in_bytes));
//	if (error != cudaSuccess)
//	{
//		fprintf(stderr, " Error : %s \n", cudaGetErrorString); //cuda get error from function
//	}
//	error = (cudaMalloc((int**)&d_c, size_in_bytes));
//	if (error != cudaSuccess)
//	{
//		fprintf(stderr, " Error : %s \n", cudaGetErrorString); //cuda get error from function
//	}
//	error = (cudaMalloc((int**)&gpu_result_k, size_in_bytes));
//	if (error != cudaSuccess)
//	{
//		fprintf(stderr, " Error : %s \n", cudaGetErrorString); //cuda get error from function
//	}
//
//	auto htod_start = std::chrono::high_resolution_clock::now();
//	cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, h_b, size_in_bytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_c, h_c, size_in_bytes, cudaMemcpyHostToDevice);
//	auto htod_end = std::chrono::high_resolution_clock::now();
//
//	//GPU config
//	dim3 block(block_size);
//	dim3 grid((array_size / block.x) + 1);
//
//	//CPU launch function
//	auto cpu_start = std::chrono::high_resolution_clock::now();
//	add_3_cpu(h_a, h_b, h_c, cpu_result, array_size);
//	auto cpu_end = std::chrono::high_resolution_clock::now();
//
//	//GPU launch kernel
//	auto gpu_start = std::chrono::high_resolution_clock::now();
//	add_3_gpu << <grid, block >> > (d_a, d_b, d_c, gpu_result_k, array_size);
//	auto gpu_end = std::chrono::high_resolution_clock::now();
//	cudaDeviceSynchronize();
//
//	auto dtoh_start = std::chrono::high_resolution_clock::now();
//	cudaMemcpy(gpu_result, gpu_result_k, size_in_bytes, cudaMemcpyDeviceToHost);
//	auto dtoh_end = std::chrono::high_resolution_clock::now();
//
//	compare_arrays(cpu_result, gpu_result, array_size);
//	cudaFree(d_a);
//	cudaFree(d_b);
//	cudaFree(d_c);
//	cudaFree(gpu_result);
//
//	free(h_a);
//	free(h_b);
//	free(h_c);
//	free(cpu_result);
//
//	printf("Sum function execution time on CPU: %d micro sec \n", std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
//	printf("Sum function execution time on GPU: %d micro sec \n", std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count());
//	printf("Memory transfer from host to device, Upload time: %d micro sec \n", std::chrono::duration_cast<std::chrono::microseconds>(htod_end - htod_start).count());
//	printf("Memory transfer from device to host, Download time: %d micro sec \n", std::chrono::duration_cast<std::chrono::microseconds>(dtoh_end - dtoh_start).count());
//
//	std::cout << "Finished" << std::endl;
//	return 0;
//}