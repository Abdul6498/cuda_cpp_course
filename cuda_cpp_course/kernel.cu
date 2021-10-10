
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//16x16 thread divided into 4 different threads of 8x8 threads. THis behaves like a nested loop with 4 inner loops

__global__ void print_threads() {

    printf("threadIdx.x : %d, threadIdx.y : %d, threadidx.z : %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    int nx, ny;
    //total number of threads in x and y direction
    nx = 16;
    ny = 16;

    //size of each block in a bigger 2d array/block
    dim3 block(8, 8);
    //number of block in x direction and number of in y direction in this case its 2 and 2
    dim3 grid(nx / block.x, ny / block.y);

    
    print_threads << < grid, block >> > (); //(number of blocks in x,y,z direction , size of each block)
    
    return 0;
}
