#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <fstream>

#include <curand_kernel.h>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

using namespace std;

__device__ int d_count = 0;

__global__ void colourMax (int *vertexArray, int *neighbourArray, int *numbers, int n, int m, int *colouring, int currentColour){

	int i= blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i>=n){
		return;
	}
	
	if (colouring[i]!=0){
		return;
	}
	
	int myMax = numbers[i];

//	printf("I am node %d with value %d\n", i+1, myMax);
	
	int start = -1, stop = -1;
	
	start = vertexArray[i];
	
	stop = vertexArray[i+1];
	
	for (int j=start; j<stop; j++){
		
//		printf("My neighbour %d with value %d from %d \n", neighbourArray[j], numbers[neighbourArray[j]-1], i+1);
	
		int neighbour = neighbourArray[j]-1;
		
		if (colouring[neighbour]==0 && numbers[neighbour] >= myMax){
			if (numbers[neighbour] == myMax){
				if (i < neighbour){
					continue;
				}
			}
			
			return;
		}
	}
	
	colouring[i] = currentColour;
	
	atomicAdd(&d_count, 1);
}

__global__ void setup_kernel (curandState * state, unsigned long seed ){

    int i= blockDim.x * blockIdx.x + threadIdx.x;

    curand_init (seed, i, 0, &state[i]);
} 

__global__ void randomNumbering (curandState* globalState, int *degreeCount, int n, int limit){

	int i= blockDim.x * blockIdx.x + threadIdx.x;
	
	curandState localState = globalState[i];
    float RANDOM = curand_uniform( &localState );
    globalState[i] = localState;
    
    RANDOM *= (limit - 1 + 0.999999);
    RANDOM += 1;
	
	degreeCount[i] = (int) RANDOM;
}

__global__ void degreeCalc (int *vertexArray, int *neighbourArray, int *degreeCount, int n, int m){

	int i= blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i>=n){
		return;
	}
	
	
	int start = -1, stop = -1;
	int diff=0;
	
	start = vertexArray[i];
	
	stop = vertexArray[i+1];

	diff = stop-start;
		
	degreeCount[i]=diff;
}

void edgesPrint (int vertexArray[], int neighbourArray[], int n, int m){ 

	for (int i=0; i<n-1; i++){
		for (int j = vertexArray[i]; j < vertexArray[i+1]; ++j){

				cout<<"e "<<i+1<<" "<<neighbourArray[j]<<endl;
				/* code */
		}
	}

	for (int j = vertexArray[n-1]; j < m; ++j)
	{
			cout<<"e "<<n<<" "<<neighbourArray[j]<<endl;
				/* code */
		
	}
}

int main(int argc, char const *argv[])
{
	string a, b;
	int n, m;
	
	cin>>n>>m;
	
	ofstream fout;
	fout.open("output2.txt",ios::app);
	
	int *h_count = new int;

	int *h_vertexArray = new int [n+1];
	int *h_neighbourArray = new int [2*m];
	int *h_degreeCount = new int [n];
	int *h_colour = new int [n];
	
	int *d_vertexArray = NULL;
    cudaMalloc((void **)&d_vertexArray, (n+1)*sizeof(int));
    	
    int *d_neighbourArray = NULL;
    cudaMalloc((void **)&d_neighbourArray, 2*m*sizeof(int));
    	
    int *d_colour = NULL;
    cudaMalloc((void **)&d_colour, (n)*sizeof(int));
    cudaMemset((void *)d_colour, 0, (n)*sizeof(int));
    	
    int *d_degreeCount = NULL;
    cudaMalloc((void **)&d_degreeCount, (n)*sizeof(int));
    cudaMemset((void *)d_degreeCount, 0, (n)*sizeof(int));
    	
    curandState* devStates;
    cudaMalloc ( &devStates, n*sizeof( curandState ) );
    	
	for (int i = 0; i < n+1; ++i)
	{
		h_vertexArray[i]=2*m;
	}

	int NSlast = 0;
	int NSoffset = 0;
	int NSprev=0;
	
	
	for (int i=0; i<2*m; i++){
		int start, end;
		cin>>start>>end;
		
		for (int j=NSlast+1; j<start; j++){
			h_vertexArray[j-1]=NSoffset;
			
		}
		
		if (NSprev!=start){
			NSlast=start;
			h_vertexArray[start-1]=NSoffset;
			NSprev=start;
		}
		
		h_neighbourArray[NSoffset]=end;
		NSoffset++;
		
	}
	
	cudaEvent_t start, stop;
	float timeNew;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	
	cudaMemcpy(d_vertexArray, h_vertexArray, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbourArray, h_neighbourArray, 2*m*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&timeNew, start, stop);
	fout<<timeNew<<"\t";
	
	int threadsPerBlock = 512;
	int blocksPerGrid = (n + threadsPerBlock -1)/threadsPerBlock;
	
	degreeCalc<<<blocksPerGrid, threadsPerBlock>>>(d_vertexArray, d_neighbourArray, d_degreeCount, n, m);
	
	thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_degreeCount);
  	int maxDegree = *(thrust::max_element(d_ptr, d_ptr + n));

	cout<<"Max = "<<maxDegree<<endl;
	
	cudaEventRecord(start, 0);
	
	setup_kernel <<<blocksPerGrid, threadsPerBlock>>> ( devStates, time(NULL) );
	
	randomNumbering<<<blocksPerGrid, threadsPerBlock>>>(devStates, d_degreeCount, n, n);

//	cudaMemcpy(h_degreeCount, d_degreeCount, n*sizeof(int), cudaMemcpyDeviceToHost);

//	cout<<"Random numbers: "<<endl;
//	
//	for (int i=0; i<n; i++){
//		cout<<h_degreeCount[i]<<endl;
//	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&timeNew, start, stop);
	fout<<timeNew<<"\t";
	
	int colourCount = 1;
		
	cudaEventRecord(start, 0);
	
	while (1){
		colourMax<<<blocksPerGrid, threadsPerBlock>>>(d_vertexArray, d_neighbourArray, d_degreeCount, n, m, d_colour, colourCount);
	
		cudaMemcpyFromSymbol(h_count, d_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
		
//		cout<<"H Count = "<<*h_count<<"at colour: "<<colourCount<<endl;
		
		if (*h_count == n){
			break;
		}
		
		colourCount++;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&timeNew, start, stop);
	fout<<timeNew<<"\t"<<colourCount<<"\t";
	
	cudaEventRecord(start, 0);
	
  	cudaMemcpy(h_colour, d_colour, n*sizeof(int), cudaMemcpyDeviceToHost);
  	
  	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&timeNew, start, stop);
	fout<<timeNew<<"\t";
  	
  	thrust::device_ptr<int> c_ptr = thrust::device_pointer_cast(d_colour);
  	int maxColour = *(thrust::max_element(c_ptr, c_ptr + n));

	cout<<"Max Colour = "<<maxColour<<endl;
  	
  	fout<<maxColour<<"\n";
  	
//  	cout<<"Colour numbers: "<<endl;
//	
//	for (int i=0; i<n; i++){
//		cout<<h_colour[i]<<endl;
//	}

	delete h_count;		
	delete[] h_vertexArray;
	delete[] h_neighbourArray;
	delete[] h_degreeCount;
	delete[] h_colour;
	
	cudaFree(d_neighbourArray);
	cudaFree(d_vertexArray);
	cudaFree(d_degreeCount);
	cudaFree(d_colour);
	
	cudaDeviceReset();
	return 0;
}
