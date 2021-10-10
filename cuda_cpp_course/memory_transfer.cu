#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>



int main() {


	cudaDeviceReset();

	printf("Finished");
	return 0;
}