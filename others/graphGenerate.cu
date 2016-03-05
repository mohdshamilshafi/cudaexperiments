#include <iostream>
#include <cstdlib>

using namespace std;

__global__ void graphGenerate (float *a, float *b, int n){
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


int main(){
	
	int n = 50000;
	size_t size = n * sizeof(float);
	
	float *h_a = new float[n];
	float *h_b = new float[n];
	float *h_c = new float[n];
	
	
	float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);
    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);
    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);
    
    //cudaMemcpy(d_A, h_a, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_B, h_b, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock -1)/threadsPerBlock;
    
    
    vectorValue<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, n);
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    
    cudaMemcpy(h_a, d_A, size, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_b, d_B, size, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_c, d_C, size, cudaMemcpyDeviceToHost);
    
    for (int i=0; i<n; i++){
    	cout<<"A["<<i<<"] = "<<h_a[i]<<endl;
    	cout<<"B["<<i<<"] = "<<h_b[i]<<endl;
    	cout<<"C["<<i<<"] = "<<h_c[i]<<endl;
    }

    cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	// Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
			
	return 0;
}
