#include "Header.h"

//__global__ void dynamic_parallelism(int size, int depth)
//{
//	printf("Depth : %d, tid : %d, blockIdx.x: %d \n", depth, threadIdx.x, blockIdx.x);
//
//	//stop condition
//	if (size == 1)
//		return;
//
//	if (threadIdx.x == 0)
//	{
//		dynamic_parallelism << <1, size / 2 >> > (size/2, depth+1);
//	}
//}

//int main()
//{
//
//	dynamic_parallelism << <1, 8 >> > (8, 0);
//	cudaDeviceSynchronize();
//	cudaDeviceReset();
//	printf("Finished \n");
//	return 0;
//}