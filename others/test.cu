#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace std;

__global__ void matrixColour (float *a, float *b, int n){
	int j= blockDim.x * blockIdx.x + threadIdx.x;
	
	printf("Block = %d ; Thread = %d \n",blockIdx.x+1, threadIdx.x+1);
	
//	if(j<n){
//		for (int i=0; i<n; i++){
////			printf("Block = %d ; Thread = %d ; i = %d ; %f\n",blockIdx.x+1, j+1,i+1,b[i]);
//			if (a[j*n+i]==1){
//				if (b[j]==b[i]){
//					b[j]=-1;
//					break;
//				}
//			}
//		}
//	}
	
	int *colour = new int[n];
	
	memset(colour, 0, n*sizeof(int));
	
	if (j<n){
		for (int i=0; i<n; i++){
			//printf("Thread = %d ; i = %d ; %f\n",j+1,i+1,b[i]);
			printf("Block = %d ; Thread = %d First For i = %d\n",blockIdx.x+1, threadIdx.x+1, i+1);
			if (a[j*n+i]==1 && b[i]!=-1){
				colour[(int)b[i]]=1;
			}
			
			
//			if (i==j){
//				//atomicAdd(&b[i],1.0f);
//				b[i]+=1.0f;
//			}
		}
		
		for (int i=0; i<n; i++){
			
			if (colour[i]==0){
				printf("Block = %d ; Thread = %d Second For i = %d\n",blockIdx.x+1, threadIdx.x+1, i+1);
				atomicAdd(&b[j],(float)i-b[j]);
				break;
			}
		}
		
		
//		for (int i=0; i<n; i++){
//			printf("Third Block = %d ; ThreadId = %d ; Thread = %d ; i = %d ; %f\n",blockIdx.x+1, threadIdx.x+1, j+1,i+1,b[i]);		
//		}
		
	}
	
	
	

//	printf("I am thread no: %d from blocknumber: %d\n", threadIdx.x, blockIdx.x);
	
	//b[j] = j+1;
	
	
}

int main(){
	
	int noNodes = 64;
	int n = noNodes*noNodes;
	
	size_t size = n * sizeof(float);
	size_t sizeColouring = noNodes * sizeof(float);
	
	float *h_matrix = new float[n];
	float *h_colouring = new float[noNodes];
	float *h_colouring1 = new float[noNodes];
	
	
	
	
	int k=100;
	
	for (int i=0; i<noNodes; i++){
		h_colouring[i]=-1;
		h_colouring1[i]=-1;
		for (int j=0; j<noNodes; j++){
			if (i==j){
				h_matrix[noNodes*i+j]=0;
			}
			
			else{
				h_matrix[noNodes*i+j]=rand()%2;
			}
		}
	}
	
//	for (int i=0; i<noNodes; i++){
//	
//		for (int j=0; j<noNodes; j++){
//		
//			cout<<h_matrix[noNodes*i+j]<<" ";	
//		}
//		
//		cout<<endl;
//		
//	}
	
	float *d_matrix = NULL;
    cudaMalloc((void **)&d_matrix, size);
    
    float *d_colouring = NULL;
    cudaMalloc((void **)&d_colouring, sizeColouring);
    
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_colouring, h_colouring, sizeColouring, cudaMemcpyHostToDevice);
	
//	for (int i=0; i<noNodes; i++){
//		cout<<"Back Home i = "<<i+1<<" ; "<<h_colouring[i]<<endl;
//	}

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (noNodes + threadsPerBlock -1)/threadsPerBlock;
    
    cout<<"Blocks "<<threadsPerBlock<<" "<<blocksPerGrid<<endl;
    
    matrixColour<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_colouring, noNodes);
    
    cudaMemcpy(h_colouring, d_colouring, sizeColouring, cudaMemcpyDeviceToHost);

	
	for (int i=0; i<noNodes; i++){
		if (h_colouring[i]!=-1)
			cout<<"Back Home i = "<<i+1<<" ; "<<h_colouring[i]<<endl;
	}
	
	bool colour[noNodes];
	
	memset(colour, 0, noNodes*sizeof(bool));
	
	for (int i=0; i<noNodes; i++){
		if (h_colouring[i]==-1){
			for (int j=0; j<noNodes; j++){
				if (h_matrix[i*noNodes+j]==1){
					if (h_colouring[j] >= 0)	
						colour[(int)h_colouring[j]]=true;
				}
			}
			
			for (int j=0; j<noNodes; j++){
				if (colour[j]==false){
					h_colouring[i]=j;
					break;
				}
			}
			memset(colour, 0, noNodes*sizeof(bool));
		}
	}
	
//	for (int i=0; i<noNodes; i++){
//		cout<<"Back Home i = "<<i+1<<" ; "<<h_colouring1[i]<<endl;
//	}
	
	for (int i=0; i<noNodes; i++){
		if (h_colouring1[i]==-1){
			for (int j=0; j<noNodes; j++){
				if (h_matrix[i*noNodes+j]==1){
					if (h_colouring1[j] >= 0)	
						colour[(int)h_colouring1[j]]=true;
				}
			}
			
			for (int j=0; j<noNodes; j++){
				if (colour[j]==false){
					h_colouring1[i]=j;
					break;
				}
			}
			memset(colour, 0, noNodes*sizeof(bool));
		}
	}


//	for (int i=0; i<noNodes; i++){
//		
//		cout<<"Back Home i = "<<i+1<<" ; "<<h_colouring[i]<<endl;
//	}


//	for (int i=0; i<noNodes; i++){
//		cout<<"Back Home i = "<<i+1<<" ; "<<h_colouring1[i]<<endl;
//	}
//	
    cudaFree(d_matrix);
	cudaFree(d_colouring);
	
    free(h_colouring);
    free(h_matrix);

    cudaDeviceReset();
			
	return 0;
}
