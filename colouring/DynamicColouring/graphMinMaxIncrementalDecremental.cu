#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <curand_kernel.h>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

using namespace std;

#define bucketLimit 600

__device__ int d_count = 0;


__global__ void propagationColouring (int *vertexArray, int *neighbourArray, int n, int m, int *colouring, int start, int end, int maxColour){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= n){
		return;
	}
	
}

__global__ void decrementalColouring (int *vertexArray, int *neighbourArray, int n, int m, int *colouring, int start, int end, int maxColour){
	
	int i = threadIdx.x;
	
	int startStart, startStop;
	int me, you;
	
	if (i==0){
		me = start;
		you = end;
		
//		printf("I am %d and my me is %d and my you is %d", i, me, you);
	}
	else{
		me = end;
		you = start;
		
//		printf("I am %d and my me is %d and my you is %d", i, me, you);
	}
	
	startStart = vertexArray[me-1];
	
//	printf("I am %d and my startStart is %d", i, startStart);
	
	if (me==n){	
		startStop = 2*m;
	}
	
	else{
		startStop = vertexArray[me];
	}
	
	
//	printf("I am %d and my startStop is %d", i, startStop);
		
	for (int j=startStart; j<startStop; j++){
		if (neighbourArray[j]==you){
			neighbourArray[j]=0;
			break;
		}
	}	
	
	__syncthreads();
	
    int colours=0;
	
	bool bucket[bucketLimit];
	
	int colouringLimit = colouring[me-1];
	
	for (int j=0; j<colouringLimit-1; j++){
		bucket[j]=true;
	}	
	
	for (int j=startStart; j<startStop; j++){
		if (neighbourArray[j]==0){
			continue;
		}
		
		int bucketIndex = colouring[neighbourArray[j]-1]; 
		
		if (bucketIndex < colouringLimit){
			bucket[bucketIndex-1] = false;
		}
		
	}
	
	for (int j=0; j<colouringLimit; j++){
		if(bucket[j]){
			colours=j+1;
			break;
		}
	}
	
	if (!colours){
		return;
	}
	
	colouring[me-1]=colours;
}

__global__ void incrementalColouring (int *vertexArray, int *neighbourArray, int n, int m, int *colouring, int start, int end, int maxColour){
	
	int i = threadIdx.x;
	
	int startStart, startStop;
	int me, you;
	
	if (i==0){
		me = start;
		you = end;
	}
	else{
		me = end;
		you = start;
	}
	
	startStart = vertexArray[me-1];
	
	if (me==n){	
		startStop = 2*m;
	}
	
	else{
		startStop = vertexArray[me];
	}
		
	for (int j=startStart; j<startStop; j++){
		if (neighbourArray[j]==0){
			neighbourArray[j]=you;
			break;
		}
	}	
	
	__syncthreads();
	
	if (colouring[start-1]!=colouring[end-1]){
		return;
	}
	
//	if (i==0)
//	printf("%d and %d Conflict\n", start, end);
	
	__shared__ int colours[2];
	
	colours[i]=0;
	
//	if (i==0)
//	printf("I am %d and %d and %d\n", i, colours[i], colours[1-i]);
	
	bool bucket[bucketLimit];
	
	for (int j=0; j<maxColour; j++){
		bucket[j]=true;
	}
	
//	if (i==0){
//		printf("%d %d", startStart, startStop);
//		
//		for (int j=startStart; j<startStop; j++){
//		
//			printf("clo %d\n", neighbourArray[j]);
//		
//			if (neighbourArray[j]!=0){
//				printf("clocli %d\n", colouring[neighbourArray[j]-1]);
//			}
//		}
//	}
	
	for (int j=startStart; j<startStop; j++){
		if (neighbourArray[j]==0){
			continue;
		}
		
		bucket[colouring[neighbourArray[j]-1]-1] = false;
		
//		if (i==0)
//		printf("buvket clo %d and %d and %d\n", neighbourArray[j]-1, colouring[neighbourArray[j]-1], bucket[colouring[neighbourArray[j]-1]-1]);
	}
	
	for (int j=0; j<maxColour; j++){
		if(bucket[j]){
			colours[i]=j+1;
//			printf("%d ashhas \t", j+1);	
			break;
		}
	}
	
//	if (i==0)
//	for (int j=0; j<maxColour; j++){
//		printf("%d \t",bucket[j]);
//	}
	
//	if (i==0){
//		printf("\n");
//	}
	
	
	__syncthreads();
	
//	printf("%d and %d Conflict  new colour min %d \n", start, end, colours[i]);
	
	// Possible issue: There could be a number inbetween the smallest equal guess and the current colour.
	
	if (colours[i]==colours[1-i]){
		if (colours[i]<colouring[me-1]){
			if(i==0){
				colouring[me-1]=colours[i];
			}
		}
	
		else{
			if (i==1){
				colouring[me-1]=colours[i];
			}
		}
	}
	
	else{
		if (colours[i]<colouring[me-1]){
			colouring[me-1]=colours[i];
		}
		
		else{
			if (colours[i]<colours[1-i]){
				colouring[me-1]=colours[i];
			}
		}
	}
	
	__syncthreads();
	
//	if (i==0){
//		for (int j=0; j<n; j++){
//			printf("%d ", colouring[j]);
//		}
//		printf("\n");
//	}
	
	
}

__global__ void colourMinMax (int *vertexArray, int *neighbourArray, int *numbers, int n, int m, int *colouring, int currentColour){

	int i= blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i>=n){
		return;
	}
	
	if (colouring[i]!=0){
		return;
	}
	
	int myValue = numbers[i];

//	printf("I am node %d with value %d\n", i+1, myMax);
	
	int start = -1, stop = -1;
	
	start = vertexArray[i];
	
	if (i==n-1){	
		stop = 2*m;
	}
	
	else{
		stop = vertexArray[i+1];
	}
	
	bool max = true, min = true;
	
	for (int j=start; j<stop; j++){
		
//		printf("My neighbour %d with value %d from %d \n", neighbourArray[j], numbers[neighbourArray[j]-1], i+1);
	
		int neighbour = neighbourArray[j];
		
		if (neighbour==0){
			continue;
		}
		
		neighbour--;
		
		if (max && colouring[neighbour]==0 && numbers[neighbour] >= myValue){
			if (numbers[neighbour] == myValue){
				if (i < neighbour){
					continue;
				}
			}
			
			max=false;
			
			if (!min){
				return;
			}
		}
		
		if (min && colouring[neighbour]==0 && numbers[neighbour] <= myValue){
			if (numbers[neighbour] == myValue){
				if (i > neighbour){
					continue;
				}
			}
			
			min=false;
			
			if (!max){
				return;
			}
		}
	}
	
	if (max){
		colouring[i] = currentColour;	
	}
	else if (min){
		colouring[i] = currentColour+1;
	}
	
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
	
	if (i==n-1){	
		stop = 2*m;
	}
	
	else{
		stop = vertexArray[i+1];
	}

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
	int n, m;
	
	cin>>n>>m;
	
	int h_maxColour;
	
	int *h_count = new int;

	int *h_vertexArray = new int [n];
	int *h_neighbourArray = new int [2*m];
	int *h_degreeCount = new int [n];
	int *h_colour = new int [n];
	
	int *d_vertexArray = NULL;
    cudaMalloc((void **)&d_vertexArray, n*sizeof(int));
    	
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
    
    int offset = 0;
    
    vector<int> startArray, stopArray;
    
    cin>>h_maxColour;
    
	for (int i = 0; i < n; ++i)
	{
		h_vertexArray[i]=offset;
		int degree;
		cin>>degree;
		offset+=degree;
	}

	for (int i = 0; i < 2*m; ++i)
	{
		h_neighbourArray[i]=0;
	}

	for (int i = 0; i < m; ++i)
	{
		int start;
		int end;

		cin>>start>>end;

		double r = ((double) rand() / (RAND_MAX));
		
		if (r<=0.5){
			int startStart, startStop, stopStart, stopStop;
			
			startStart = h_vertexArray[start-1];
	
			if (start==n){	
				startStop = 2*m;
			}
	
			else{
				startStop = h_vertexArray[start];
			}
			
			
			stopStart = h_vertexArray[end-1];
	
			if (end==n){	
				stopStop = 2*m;
			}
	
			else{
				stopStop = h_vertexArray[end];
			}
			
			for (int j=startStart; j<startStop; j++){
				if (h_neighbourArray[j]==0){
					h_neighbourArray[j]=end;
					break;
				}
			}
			
			for (int j=stopStart; j<stopStop; j++){
				if (h_neighbourArray[j]==0){
					h_neighbourArray[j]=start;
					break;
				}
			}
		}
		
		else{
			startArray.push_back(start);
			stopArray.push_back(end);
		}

	}
	
//	for (int i=0; i<n; i++){
//		cout<<h_vertexArray[i]<<" ";
//	}
//	
//	cout<<endl;
//	
//	for (int i=0; i<2*m; i++){
//		cout<<h_neighbourArray[i]<<" ";
//	}
//	
//	cout<<endl;
	
	cudaMemcpy(d_vertexArray, h_vertexArray, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbourArray, h_neighbourArray, 2*m*sizeof(int), cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 512;
	int blocksPerGrid = (n + threadsPerBlock -1)/threadsPerBlock;

	setup_kernel <<<blocksPerGrid, threadsPerBlock>>> ( devStates, time(NULL) );
	
	randomNumbering<<<blocksPerGrid, threadsPerBlock>>>(devStates, d_degreeCount, n, n);

	cudaMemcpy(h_degreeCount, d_degreeCount, n*sizeof(int), cudaMemcpyDeviceToHost);

//	cout<<"Random numbers: "<<endl;
//	
//	for (int i=0; i<n; i++){
//		cout<<h_degreeCount[i]<<endl;
//	}

	int colourCount = 1;
	
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	
//	cout<<"Worklist: "<<endl;
//	
//	for	(int i=0; i<startArray.size(); i++){
//		cout<<startArray[i]<<" "<<stopArray[i]<<endl;
//	}
//	
	while (1){
		colourMinMax<<<blocksPerGrid, threadsPerBlock>>>(d_vertexArray, d_neighbourArray, d_degreeCount, n, m, d_colour, colourCount);
	
		cudaMemcpyFromSymbol(h_count, d_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
		
		cout<<"H Count = "<<*h_count<<"at colour: "<<colourCount<<endl;
		
		if (*h_count == n){
			break;
		}
		
		colourCount+=2;
	}
	
	colourCount++;
	
	cout<<"Size: "<<startArray.size()<<endl;
	
	for (int i=0; i<startArray.size(); i++){
	
		cout<<"New added edge: "<<startArray[i]<<" "<<stopArray[i]<<endl;
		
		incrementalColouring<<<1, 2>>>(d_vertexArray, d_neighbourArray, n, m, d_colour, startArray[i], stopArray[i], h_maxColour);
		
		cudaDeviceSynchronize();
		
	}
	
	for (int i=0; i<startArray.size(); i++){
	
		cout<<"Deleted edge: "<<startArray[i]<<" "<<stopArray[i]<<endl;
		
		decrementalColouring<<<1, 2>>>(d_vertexArray, d_neighbourArray, n, m, d_colour, startArray[i], stopArray[i], h_maxColour);
		
		cudaDeviceSynchronize();
		
	}
	
	cout<<"Shamil"<<endl;
	
	cudaMemcpy(h_colour, d_colour, n*sizeof(int), cudaMemcpyDeviceToHost);
  	
  	thrust::device_ptr<int> c_ptr = thrust::device_pointer_cast(d_colour);
  	int maxColour = *(thrust::max_element(c_ptr, c_ptr + n));

	cout<<"Max Colour = "<<maxColour<<endl;
  	
  	cout<<"Colour numbers: "<<endl;
	
	
	
	for (int i=0; i<n; i++){
		cout<<h_colour[i]<<endl;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&time, start, stop);
	cout<<"Time for the kernel: "<<time<<" ms"<<endl;
	
	cudaMemcpy(h_vertexArray, d_vertexArray, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_neighbourArray, d_neighbourArray, 2*m*sizeof(int), cudaMemcpyDeviceToHost);
	
//	for (int i=0; i<n; i++){
//		cout<<h_vertexArray[i]<<" ";
//	}
//	
//	cout<<endl;
//	
//	for (int i=0; i<2*m; i++){
//		cout<<h_neighbourArray[i]<<" ";
//	}
//	
//	cout<<endl;

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
