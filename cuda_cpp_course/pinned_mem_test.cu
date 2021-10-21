#include "Header.h"

//int main()
//{
//	//memory size = 128 Mbs
//	int isize = 1 << 25;
//	int nbytes = isize * sizeof(int);
//
//	//allocate host memory
//	//float* h_a = (float*)malloc(nbytes);
//	float* h_a;
//	cudaMallocHost((float**)&h_a, nbytes);
//	//allocate the device memory
//	float* d_a;
//	cudaMalloc((float**)&d_a, nbytes);
//
//	// intialize host memory
//	for (int i = 0; i < isize; i++)
//		h_a[i] = 7;
//
//	//transfer mem from host to device
//	cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
//
//	//transfer data from device to host
//	cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);
//
//	//free memory
//	cudaFree(d_a);
//	cudaFreeHost(h_a);
//	//free(h_a);
//
//	cudaDeviceReset();
//	printf("Finished \n");
//	return 0;
//}