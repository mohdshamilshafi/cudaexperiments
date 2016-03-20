#include <iostream>
#include <cstdlib>
#include <cstdio>

#include <curand_kernel.h>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#define N 10

using namespace std;

struct node{
	int base;
	int data;
	
	node *next;
};

__device__ node *head[N];


__global__ void print_kernel()
{
    int i= blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= N){
		return;
	}
	
	node *temp = head[i];
	
	while (temp){
		printf("%d and %d from %d\n", temp->base, temp->data, i);
		
		temp = temp->next;
	}
	
}



__global__ void setup_kernel()
{
    int i= blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= N){
		return;
	}
	
	head[i]=NULL;
	
	node *end = head[i];
	
	for (int j=i; j<i+5; j++){
		node *temp = new node();
		temp->base = j;
		temp->data = j;
		temp->next = NULL;
		
		if (end){
			end->next = temp;
			end = end->next;
		}
		else{
			head[i]=temp;
			end = head[i];
		}
	}
	
}

int main(int argc, char const *argv[])
{
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;
	
	setup_kernel<<<blocksPerGrid, threadsPerBlock>>>();
	
	print_kernel<<<blocksPerGrid, threadsPerBlock>>>();
			
	cudaDeviceReset();
	return 0;
}
