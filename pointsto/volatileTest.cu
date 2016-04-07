#include <iostream>
#include <cstdlib>
#include <cstdio>


#include <curand_kernel.h>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

using namespace std;

__device__ int sum = 1;

__global__ void degreeCalc (int *array){
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i>=1000000){
		return;
	}
	
	sum+=array[i];
	
//	if (i==999999){
//		printf("%d", sum);
//	}
}


int main(int argc, char const *argv[])
{
	/* code */
	
	int n = 1000000;
	
	int *h_array = new int [n];
	
	int *h_sum = new int;
	
	int *d_array = NULL;
    cudaMalloc((void **)&d_array, n*sizeof(int));
    	
    	
	for (int i = 0; i < n; ++i)
	{
		/* code */
		h_array[i]=1;
	}

	cudaMemcpy(d_array, h_array, n*sizeof(int), cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 512;
	int blocksPerGrid = (n + threadsPerBlock -1)/threadsPerBlock;
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
 
	cudaEventRecord(start, 0);
 		
	degreeCalc<<<blocksPerGrid, threadsPerBlock>>>(d_array);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaMemcpyFromSymbol(h_sum, sum, sizeof(int), 0, cudaMemcpyDeviceToHost);

	cout<<*h_sum<<endl;

	// Retrieve result from device and store it in host array
	cudaEventElapsedTime(&time, start, stop);
	cout<<"Time for the kernel: "<<time<<" ms"<<endl;

	
	delete[] h_array;
	cudaFree(d_array);
	
	cudaDeviceReset();
	return 0;
}
