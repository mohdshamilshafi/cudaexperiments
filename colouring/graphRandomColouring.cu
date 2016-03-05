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

__global__ void setup_kernel (curandState * state, unsigned long seed )
{
    int i= blockDim.x * blockIdx.x + threadIdx.x;

    curand_init (seed, i, 0, &state[i]);
} 

__global__ void randomColouring (curandState* globalState, int *degreeCount, int n, int limit){

	int i= blockDim.x * blockIdx.x + threadIdx.x;
	
	curandState localState = globalState[i];
    	float RANDOM = curand_uniform( &localState );
    	globalState[i] = localState;
    	
    	RANDOM *= (limit - 1 + 0.999999);
    	RANDOM += 1;
	
	degreeCount[i] = (int) RANDOM;
}

__global__ void conflictDetection (int *vertexArray, int *neighbourArray, int *degreeCount, int n, int m, int *detectConflict){

	int i= blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i>=n){
		return;
	}
	
	int myColour = degreeCount[i];
	
	int start = -1, stop = -1;
	
	start = vertexArray[i];
	
	if (i==n-1){	
		stop = m;
	}
	
	else{
		stop = vertexArray[i+1];
	}
	
	for (int j=start; j<stop; j++){
		if (degreeCount[neighbourArray[j]-1] == myColour){
			if (i < neighbourArray[j]-1){
				if (detectConflict[i]!=1){
					detectConflict[i]=1;
				}
			}
			else if (detectConflict[neighbourArray[j]-1]!=1){
				detectConflict[neighbourArray[j]-1]=1;
			}
		}
	}
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
		stop = m;
	}
	
	else{
		stop = vertexArray[i+1];
	}

	diff = stop-start;
		
	atomicAdd(&degreeCount[i], diff);
	
	for (int j=start; j<stop; j++){
		atomicAdd(&degreeCount[neighbourArray[j]-1], 1);
	}
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
	/* code */
	string a, b;
	int n, m;
	
	cin>>a>>b>>n>>m;
	
	// cout<<a<<" "<<b<<" "<<n<<" "<<m<<endl;

	int h_vertexArray[n];
	int h_neighbourArray[m];
	int h_degreeCount[n];
	int h_detectConflict[n];
	
	int *d_vertexArray = NULL;
    	cudaMalloc((void **)&d_vertexArray, n*sizeof(int));
    	
    	int *d_neighbourArray = NULL;
    	cudaMalloc((void **)&d_neighbourArray, m*sizeof(int));
    	
    	int *d_detectConflict = NULL;
    	cudaMalloc((void **)&d_detectConflict, (n)*sizeof(int));
    	cudaMemset((void *)d_detectConflict, 0, (n)*sizeof(int));
    	
    	int *d_degreeCount = NULL;
    	cudaMalloc((void **)&d_degreeCount, (n)*sizeof(int));
    	cudaMemset((void *)d_degreeCount, 0, (n)*sizeof(int));
    	
    	curandState* devStates;
    	cudaMalloc ( &devStates, n*sizeof( curandState ) );
    	
	for (int i = 0; i < n; ++i)
	{
		/* code */
		h_vertexArray[i]=m;
	}

	int offset = 0;

	int current = 0;
	int mark = 1;

	for (int i = 0; i < m; ++i)
	{
		/* code */
		char c;
		int start;
		int end;

		cin>>c>>start>>end;

		if (start!=mark){ 

			if (start == mark+1 && h_vertexArray[mark-1]!=m){ 

			}

			else{

				for (int j = mark; j<start; j++){ 
					h_vertexArray[j-1]=offset;
					// h_neighbourArray[offset]=0;
					// offset++;
				}
			}
			mark = start;

		}

		if (start==current){ 
			h_neighbourArray[offset]=end;
			offset++;
		}

		else { 
			current = start;

			h_vertexArray[current-1]=offset;

			h_neighbourArray[offset]=end;
			offset++;
		}
	}
	
	
	cudaMemcpy(d_vertexArray, h_vertexArray, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbourArray, h_neighbourArray, m*sizeof(int), cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 512;
	int blocksPerGrid = (n + threadsPerBlock -1)/threadsPerBlock;
	
	//cout<<threadsPerBlock<<" "<<blocksPerGrid<<endl;
	
	degreeCalc<<<blocksPerGrid, threadsPerBlock>>>(d_vertexArray, d_neighbourArray, d_degreeCount, n, m);
	
//	cudaMemcpy(h_degreeCount, d_degreeCount, n*sizeof(int), cudaMemcpyDeviceToHost);

//	for (int i=0; i<n; i++){
//		cout<<h_degreeCount[i]<<endl;
//	}
	
	thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_degreeCount);
  	int max = *(thrust::max_element(d_ptr, d_ptr + n));
	
//	int result = thrust::reduce(h_degreeCount, h_degreeCount + n,
//                           -1,
//                            thrust::maximum<int>());
                            
//        cout<<"Result: "<<result<<endl<<max;

//	cout<<"Max = "<<max<<endl;


	setup_kernel <<<blocksPerGrid, threadsPerBlock>>> ( devStates, time(NULL) );
	randomColouring<<<blocksPerGrid, threadsPerBlock>>>(devStates, d_degreeCount, n, max+1);

	// cudaMemcpy(h_degreeCount, d_degreeCount, n*sizeof(int), cudaMemcpyDeviceToHost);
	
	conflictDetection<<<blocksPerGrid, threadsPerBlock>>>(d_vertexArray, d_neighbourArray, d_degreeCount, n, m, d_detectConflict);
	
	thrust::device_ptr<int> d_detectConflict_ptr = thrust::device_pointer_cast(d_detectConflict);
  	int count1 = thrust::reduce(d_detectConflict_ptr, d_detectConflict_ptr + n);
  	
  	cudaMemcpy(h_detectConflict, d_detectConflict, n*sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i=0; i<n; i++){
		cout<<i+1<<": "<<h_detectConflict[i]<<endl;
	}
	
	cout<<"Count: "<<count1<<endl;

//	for (int i=0; i<n; i++){
//		if (h_degreeCount[i] == max+1){
//			cout<<"BUHAHAHAHAHAHHAHAHAHHAHA"<<endl;
//		}
//		
//		else if (h_degreeCount[i] == 1){
//			cout<<"LALLLALALALALALALALALLALA"<<endl;
//		}
//		cout<<h_degreeCount[i]<<endl;
//	}

//	for (int i=0; i<n; i++){
//		cout<<i+1<<": "<<h_detectConflict[i]<<endl;
//	}


	
	//edgesPrint(h_vertexArray, h_neighbourArray, n, m);

	//delete[] h_vertexArray;
	//delete[] h_neighbourArray;
	//delete[] h_degreeCount;
	
	cudaFree(d_neighbourArray);
	cudaFree(d_vertexArray);
	cudaFree(d_degreeCount);
	cudaFree(d_detectConflict);
	
	cudaDeviceReset();
	return 0;
}
