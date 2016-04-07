#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#include <cstdlib>
#include <cstdio>
#include <climits>

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
	unsigned int data[29];
	
	node *next;
};

__device__ node *headP[N];
__device__ node *headC[N];
__device__ node *headL[N];
__device__ node *headS[N];

__global__ void print_kernel1()
{
    int i= blockDim.x * blockIdx.x + threadIdx.x;

	if (i != 0){
		return;
	}
	
	node *temp = headP[i];
	
	while (temp){
		printf("Base= %d\n", temp->base);
		
		for (int k=0; k<29; k++){
			int value = temp->data[i];
			
			for (int j=0; j<32; j++){
				int bit = (value >> j) & 1;
				if (bit == 1){
					printf("Neightbour, %d\n", temp->base*928+k*32+j);
				}
			}
		}
		
		temp = temp->next;
	}
	
	temp = headC[i];
	
	while (temp){
		printf("Base= %d\n", temp->base);
		
		for (int k=0; k<29; k++){
			int value = temp->data[i];
			
			for (int j=0; j<32; j++){
				int bit = (value >> j) & 1;
				if (bit == 1){
					printf("Neightbour, %d\n", temp->base*928+k*32+j);
				}
			}
		}
		
		temp = temp->next;
	}
	
	temp = headL[i];
	
	while (temp){
		printf("Base= %d\n", temp->base);
		
		for (int k=0; k<29; k++){
			int value = temp->data[i];
			
			for (int j=0; j<32; j++){
				int bit = (value >> j) & 1;
				if (bit == 1){
					printf("Neightbour, %d\n", temp->base*928+k*32+j);
				}
			}
		}
		
		temp = temp->next;
	}
	
	temp = headS[i];
	
	while (temp){
		printf("Base= %d\n", temp->base);
		
		for (int k=0; k<29; k++){
			int value = temp->data[i];
			
			for (int j=0; j<32; j++){
				int bit = (value >> j) & 1;
				if (bit == 1){
					printf("Neightbour, %d\n", temp->base*928+k*32+j);
				}
			}
		}
		
		temp = temp->next;
	}
	
}

__global__ void print_kernel2()
{
    int i= blockDim.x * blockIdx.x + threadIdx.x;

	if (i != 1){
		return;
	}
	
	node *temp = headP[i];
	
	while (temp){
		printf("Base= %d\n", temp->base);
		
		for (int k=0; k<29; k++){
			int value = temp->data[i];
			
			for (int j=0; j<32; j++){
				int bit = (value >> j) & 1;
				if (bit == 1){
					printf("Neightbour, %d\n", temp->base*928+k*32+j);
				}
			}
		}
		
		temp = temp->next;
	}
	
	temp = headC[i];
	
	while (temp){
		printf("Base= %d\n", temp->base);
		
		for (int k=0; k<29; k++){
			int value = temp->data[i];
			
			for (int j=0; j<32; j++){
				int bit = (value >> j) & 1;
				if (bit == 1){
					printf("Neightbour, %d\n", temp->base*928+k*32+j);
				}
			}
		}
		
		temp = temp->next;
	}
	
	temp = headL[i];
	
	while (temp){
		printf("Base= %d\n", temp->base);
		
		for (int k=0; k<29; k++){
			int value = temp->data[i];
			
			for (int j=0; j<32; j++){
				int bit = (value >> j) & 1;
				if (bit == 1){
					printf("Neightbour, %d\n", temp->base*928+k*32+j);
				}
			}
		}
		
		temp = temp->next;
	}
	
	temp = headS[i];
	
	while (temp){
		printf("Base= %d\n", temp->base);
		
		for (int k=0; k<29; k++){
			int value = temp->data[i];
			
			for (int j=0; j<32; j++){
				int bit = (value >> j) & 1;
				if (bit == 1){
					printf("Neightbour, %d\n", temp->base*928+k*32+j);
				}
			}
		}
		
		temp = temp->next;
	}
	
}

//__device__ bool isBasePresent (int base, int node, int mode){
//	node *temp;
//	
//	if (mode == 0){
//		temp = headP[node];
//	}
//	
//	else if (mode == 1){
//		temp = headC[node];
//	}
//	
//	else if (mode == 2){
//		temp = headL[node];
//	}
//	
//	else{
//		temp = headS[node];
//	}
//	
//	while (temp){
//		if (temp->base == base){
//			return true;
//		}
//		
//		temp = temp->next;
//	}
//	
//	return false;
//}

__global__ void setup_kernel(int *pStartArray, int *pEndArray, int *cStartArray, int *cEndArray, int *lStartArray, int *lEndArray,
								int *sStartArray, int *sEndArray, int n, int pPos, int cPos, int lPos, int sPos)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= n){
		return;
	}
	
	printf("I am %d\n", i);
	
	headP[i]=NULL;
	headC[i]=NULL;
	headL[i]=NULL;
	headS[i]=NULL;
	
	int start = -1, end = -1;
	
	start = pStartArray[i];
	
	if (i == n-1){
		end = pPos;
	}
	
	else{
		end = pStartArray[i+1];
	}
	
	for (int j=start; j<end; j++){
		int candidate = pEndArray[j];
		
		printf("I am %d. Candidate is %d.\n", i, candidate);
		
		int base = candidate/928;
		int block = (candidate%928);
		int bit = block%32;
		block = block/32;	
		
		printf("I am %d. Base = %d, block = %d and bit = %d.\n", i, base, block, bit);
		
		bool isBasePresent=false;
		
		node *temp = headP[i];
			
		while (temp){
			if (temp->base == base){
				isBasePresent=true;
				break;
			}
				
			temp = temp->next;
		}
		
		if (isBasePresent){
			printf("I am %d and base already present.\n", i);
			temp->data[block] |= 1<<bit;
			
		}
		
		else{
			temp = new node();
			temp->base = base;
			temp->next = NULL;
			
			for (int k = 0; k<29; k++){
				temp->data[k]=0;
			}
			
			temp->data[block] |= 1<<bit;
			
			printf("I am %d and changed temp is %d.\n", i, temp->data[block]);
			
			for (int k = 0; k<29; k++){
				printf("%d ",temp->data[k]);
			}
			printf("\n");
			
			if (headP[i]==NULL){
				headP[i]=temp;
			}
			else{
				node *runner = headP[i];
				
				while (runner->next){
					runner=runner->next;
				}
				
				runner->next = temp;
			}
		}	
	}
	
	
	
	start = cStartArray[i];
	
	if (i == n-1){
		end = cPos;
	}
	
	else{
		end = cStartArray[i+1];
	}
	
	for (int j=start; j<end; j++){
		int candidate = cEndArray[j];
		
		int base = candidate/928;
		int block = (candidate%928);
		int bit = block%32;
		block = block/32;	
		bool isBasePresent=false;
		
		node *temp = headC[i];
			
		while (temp){
			if (temp->base == base){
				isBasePresent=true;
				break;
			}
				
			temp = temp->next;
		}
		
		if (isBasePresent){
			temp->data[block] |= 1<<bit;
			
		}
		
		else{
			temp = new node();
			temp->base = base;
			temp->next = NULL;
			
			for (int k = 0; k<29; k++){
				temp->data[k]=0;
			}
			
			temp->data[block] |= 1<<bit;
			
			if (headC[i]==NULL){
				headC[i]=temp;
			}
			else{
				node *runner = headC[i];
				
				while (runner->next){
					runner=runner->next;
				}
				
				runner->next = temp;
			}
		}	
	}
	
	
	start = lStartArray[i];
	
	if (i == n-1){
		end = lPos;
	}
	
	else{
		end = lStartArray[i+1];
	}
	
	for (int j=start; j<end; j++){
		int candidate = lEndArray[j];
		
		int base = candidate/928;
		int block = (candidate%928);
		int bit = block%32;
		block = block/32;	
		bool isBasePresent=false;
		
		node *temp = headL[i];
			
		while (temp){
			if (temp->base == base){
				isBasePresent=true;
				break;
			}
				
			temp = temp->next;
		}
		
		if (isBasePresent){
			temp->data[block] |= 1<<bit;
			
		}
		
		else{
			temp = new node();
			temp->base = base;
			temp->next = NULL;
			
			for (int k = 0; k<29; k++){
				temp->data[k]=0;
			}
			
			temp->data[block] |= 1<<bit;
			
			if (headL[i]==NULL){
				headL[i]=temp;
			}
			else{
				node *runner = headL[i];
				
				while (runner->next){
					runner=runner->next;
				}
				
				runner->next = temp;
			}
		}	
	}
	
	
	start = sStartArray[i];
	
	if (i == n-1){
		end = sPos;
	}
	
	else{
		end = sStartArray[i+1];
	}
	
	for (int j=start; j<end; j++){
		int candidate = sEndArray[j];
		
		int base = candidate/928;
		int block = (candidate%928);
		int bit = block%32;
		block = block/32;	
		bool isBasePresent=false;
		
		node *temp = headS[i];
			
		while (temp){
			if (temp->base == base){
				isBasePresent=true;
				break;
			}
				
			temp = temp->next;
		}
		
		if (isBasePresent){
			temp->data[block] |= 1<<bit;
			
		}
		
		else{
			temp = new node();
		temp->base = base;
			temp->next = NULL;
			
			for (int k = 0; k<29; k++){
				temp->data[k]=0;
			}
			
			temp->data[block] |= 1<<bit;
			
			if (headS[i]==NULL){
				headS[i]=temp;
			}
			else{
				node *runner = headS[i];
				
				while (runner->next){
					runner=runner->next;
				}
				
				runner->next = temp;
			}
		}	
	}
	
}

void printArray(int *a, int size){
	for (int i=0; i<size; i++){
		cout<<a[i]<<" ";
	}
	cout<<endl;
}

int main(int argc, char const *argv[])
{
	int n, m;
	
	cin>>m;
	
	int nodeCount = 0;
	
	
	int *pArray = new int[m];
	int *cArray = new int[m];
	int *lArray = new int[m];
	int *sArray = new int[m];
	
	int *pArray2 = new int[m];
	int *cArray2 = new int[m];
	int *lArray2 = new int[m];
	int *sArray2 = new int[m];
	
//	fill(pArray, pArray+m, -1);
//	fill(cArray, cArray+m, -1);
//	fill(lArray, lArray+m, -1);
//	fill(sArray, pArray+m, -1);
//	
//	fill(pArray2, pArray2+m, -1);
//	fill(cArray2, cArray2+m, -1);
//	fill(lArray2, lArray2+m, -1);
//	fill(sArray2, pArray2+m, -1);
//	
	
	int pPos = 0, cPos = 0, lPos = 0, sPos = 0;
	
	map<string, int> variablesMap;
	map<string, int>::iterator it;
	
	for (int i=0; i<m; i++){
		string a, b, temp;
		char c;
		
		cin>>a>>c>>b;
		
		int start, end;
		
		// Address
		if (b[0]=='&'){
			temp = b.substr(1);
			
			it = variablesMap.find(a);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(a,nodeCount) );
				start=nodeCount;
				nodeCount++;
			}
			else{
				start = it->second;
			}
			
			it = variablesMap.find(temp);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(temp,nodeCount) );
				end = nodeCount;
				nodeCount++;
			}
			else{
				end = it->second;
			}
			
			pArray[pPos]=start;
			pArray2[pPos]=end;
			pPos++;
		}
		
		// Load
		else if (b[0]=='*'){
			temp = b.substr(1);
			
			it = variablesMap.find(a);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(a,nodeCount) );
				start=nodeCount;
				nodeCount++;
			}
			else{
				start = it->second;
			}
			
			it = variablesMap.find(temp);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(temp,nodeCount) );
				end=nodeCount;
				nodeCount++;
			}
			else{
				end = it->second;
			}
			
			lArray[lPos]=start;
			lArray2[lPos]=end;
			lPos++;
		}
		
		// Store
		else if (a[0]=='*'){
			temp = a.substr(1);
			
			it = variablesMap.find(b);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(b,nodeCount) );
				end = nodeCount;
				nodeCount++;
			}
			else{
				end = it->second;
			}
			
			it = variablesMap.find(temp);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(temp,nodeCount) );
				start = nodeCount;
				nodeCount++;
			}
			else{
				start = it->second;
			}
			
			sArray[sPos]=start;
			sArray2[sPos]=end;
			sPos++;
		}
		
		// Copy
		else{
		
			it = variablesMap.find(a);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(a,nodeCount) );
				start=nodeCount;
				nodeCount++;
			}
			else{
				start = it->second;
			}
			
			it = variablesMap.find(b);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(b,nodeCount) );
				end = nodeCount;
				nodeCount++;
			}
			else{
				end=it->second;
			}
			
			cArray[cPos]=start;
			cArray2[cPos]=end;
			cPos++;
		}	
	}
	
//	printArray(pArray, pPos);
//	printArray(pArray2, pPos);
	thrust::sort_by_key(pArray2, pArray2 + pPos, pArray);
//	printArray(pArray, pPos);
//	printArray(pArray2, pPos);
	thrust::sort_by_key(pArray, pArray + pPos, pArray2);
//	printArray(pArray, pPos);
//	printArray(pArray2, pPos);
	
//	printArray(cArray, cPos);
//	printArray(cArray2, cPos);
	thrust::sort_by_key(cArray2, cArray2 + cPos, cArray);
//	printArray(cArray, cPos);
//	printArray(cArray2, cPos);
	thrust::sort_by_key(cArray, cArray + cPos, cArray2);
//	printArray(cArray, cPos);
//	printArray(cArray2, cPos);
	
//	printArray(lArray, lPos);
//	printArray(lArray2, lPos);
	thrust::sort_by_key(lArray2, lArray2 + lPos, lArray);
//	printArray(lArray, lPos);
//	printArray(lArray2, lPos);
	thrust::sort_by_key(lArray, lArray + lPos, lArray2);
//	printArray(lArray, lPos);
//	printArray(lArray2, lPos);
	
//	printArray(sArray, sPos);
//	printArray(sArray2, sPos);
	thrust::sort_by_key(sArray2, sArray2 + sPos, sArray);
//	printArray(sArray, sPos);
//	printArray(sArray2, sPos);
	thrust::sort_by_key(sArray, sArray + sPos, sArray2);
//	printArray(sArray, sPos);
//	printArray(sArray2, sPos);
	
	n = nodeCount;
	
	int *pStartArray = new int [n];
	int *pEndArray = new int [pPos];
	
	int *cStartArray = new int [n];
	int *cEndArray = new int [cPos];
	
	int *lStartArray = new int [n];
	int *lEndArray = new int [lPos];
	
	int *sStartArray = new int [n];
	int *sEndArray = new int [sPos];

	int NSpos = 0;
	int NSlast = -1;
	
	for (int i=0; i<n; i++){
		pStartArray[i]=pPos;
		cStartArray[i]=cPos;
		lStartArray[i]=lPos;
		sStartArray[i]=sPos;
	}
	
	for (int i=0; i<pPos; i++){
		pEndArray[i]=pArray2[i];
		
		if (pArray[i]==NSlast){
			NSpos++;
			continue;
		}
		
		for (int j=NSlast+1; j<=pArray[i]; j++){
			pStartArray[j]=NSpos;
		}
		
		NSpos++;
		NSlast = pArray[i];
		
	}
	
	
	NSpos = 0;
	NSlast = -1;
	
	for (int i=0; i<cPos; i++){
		cEndArray[i]=cArray2[i];
		
		if (cArray[i]==NSlast){
			NSpos++;
			continue;
		}
		
		for (int j=NSlast+1; j<=cArray[i]; j++){
			cStartArray[j]=NSpos;
		}
		
		NSpos++;
		NSlast = cArray[i];
		
	}
	
	NSpos = 0;
	NSlast = -1;
	
	for (int i=0; i<lPos; i++){
		lEndArray[i]=lArray2[i];
		
		if (lArray[i]==NSlast){
			NSpos++;
			continue;
		}
		
		for (int j=NSlast+1; j<=lArray[i]; j++){
			lStartArray[j]=NSpos;
		}
		
		NSpos++;
		NSlast = lArray[i];
		
	}
	
	NSpos = 0;
	NSlast = -1;
	
	for (int i=0; i<sPos; i++){
		sEndArray[i]=sArray2[i];
		
		if (sArray[i]==NSlast){
			NSpos++;
			continue;
		}
		
		for (int j=NSlast+1; j<=sArray[i]; j++){
			sStartArray[j]=NSpos;
		}
		
		NSpos++;
		NSlast = sArray[i];
		
	}

//	printArray(pArray, pPos);
//	printArray(pArray2, pPos);
//	printArray(pStartArray, n);
//	printArray(pEndArray, pPos);
//	
//	printArray(cArray, cPos);
//	printArray(cArray2, cPos);
//	printArray(cStartArray, n);
//	printArray(cEndArray, cPos);
//	
//	printArray(lArray, lPos);
//	printArray(lArray2, lPos);
//	printArray(lStartArray, n);
//	printArray(lEndArray, lPos);
//	
//	printArray(sArray, sPos);
//	printArray(sArray2, sPos);
//	printArray(sStartArray, n);
//	printArray(sEndArray, sPos);
		
	delete[] pArray;
	delete[] pArray2;
	delete[] cArray;
	delete[] cArray2;
	delete[] lArray;
	delete[] lArray2;
	delete[] sArray;
	delete[] sArray2;	
		
	int threadsPerBlock = 512;
	int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;
	
	int *d_pStartArray = NULL;
    cudaMalloc((void **)&d_pStartArray, n*sizeof(int));
    int *d_pEndArray = NULL;
    cudaMalloc((void **)&d_pEndArray, pPos*sizeof(int));
    
    int *d_cStartArray = NULL;
    cudaMalloc((void **)&d_cStartArray, n*sizeof(int));
    int *d_cEndArray = NULL;
    cudaMalloc((void **)&d_cEndArray, cPos*sizeof(int));
    
    int *d_lStartArray = NULL;
    cudaMalloc((void **)&d_lStartArray, n*sizeof(int));
    int *d_lEndArray = NULL;
    cudaMalloc((void **)&d_lEndArray, lPos*sizeof(int));
    
    int *d_sStartArray = NULL;
    cudaMalloc((void **)&d_sStartArray, n*sizeof(int));
    int *d_sEndArray = NULL;
    cudaMalloc((void **)&d_sEndArray, sPos*sizeof(int));
    
	cudaMemcpy(d_pStartArray, pStartArray, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pEndArray, pEndArray, pPos*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_cStartArray, cStartArray, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cEndArray, cEndArray, cPos*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_lStartArray, lStartArray, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lEndArray, lEndArray, lPos*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_sStartArray, sStartArray, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sEndArray, sEndArray, sPos*sizeof(int), cudaMemcpyHostToDevice);
	
	setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_pStartArray, d_pEndArray, d_cStartArray, d_cEndArray, d_lStartArray, d_lEndArray, d_sStartArray, d_sEndArray, n, pPos, cPos, lPos, sPos);
	
//	print_kernel1<<<blocksPerGrid, threadsPerBlock>>>();
//	print_kernel2<<<blocksPerGrid, threadsPerBlock>>>();
			
	cudaDeviceReset();
	
	for (it=variablesMap.begin(); it!=variablesMap.end(); ++it)
		cout << it->first << " => " << it->second << '\n';
	
	return 0;
}
