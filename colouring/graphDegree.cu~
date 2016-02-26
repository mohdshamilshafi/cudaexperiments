#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace std;

__global__ void degreeCalc (int *vertexArray, int *neighbourArray, int *degreeCount, int n, int m){
	int i= blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i<n){
		a[i]=threadIdx.x*2;
		b[i]=threadIdx.x;
	}
	
}

__global__ void vectorAddition (float *a, float *b, float *c, int n){
	int i= blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i<n){
		c[i] = a[i]+b[i];
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
	int h_degreeCount[n+1];
	
	int *d_vertexArray = NULL;
    	cudaMalloc((void **)&d_vertexArray, n*sizeof(int));
    	
    	int *d_neighbourArray = NULL;
    	cudaMalloc((void **)&d_neighbourArray, m*sizeof(int));
    	
    	int *d_degreeCount = NULL;
    	cudaMalloc((void **)&d_degreeCount, (n+1)*sizeof(int));
    	
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

	degreeCalc<<<blocksPerGrid, threadsPerBlock>>>(d_vertexArray, d_neighbourArray, d_degreeCount, n, m);
	
	cudaMemcpy(h_degreeCount, d_degreeCount, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);

	edgesPrint(h_vertexArray, h_neighbourArray, n, m);

	delete[] h_vertexArray;
	delete[] h_neighbourArray;
	delete[] h_degreeCount;
	
	cudaFree(d_neighbourArray);
	cudaFree(d_vertexArray);
	cudaFree(d_degreeCount);
	
	cudaDeviceReset();
	return 0;
}
