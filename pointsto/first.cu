#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

using namespace std;
#define B2MB(x) ((x)/(1024 * 1024))
#define WARP_SIZE 32
    
// check that the obtained solution is a subset of the desired solution. Useful when trying to 
// detect bugs (for instance, detected the 1st iteration such that the inclusion does not hold)
#define USE_INCLUSION (2)

static uint transferH2dTime = 0;
static uint transferD2hTime = 0;

static void printDeviceMemory() {
  size_t uCurAvailMemoryInBytes, uTotalMemoryInBytes;
  cudaMemGetInfo( &uCurAvailMemoryInBytes, &uTotalMemoryInBytes );
  // cout << "[host] GPU's total memory: "<< B2MB(uTotalMemoryInBytes) << " MB, free Memory: "
  //         << B2MB(uCurAvailMemoryInBytes) << " MB" << endl;    
  //if (B2MB(uCurAvailMemoryInBytes) < 3930) {
  //    cout << "Warning: there is not enough memory in your GPU to analyze all inputs." << endl;
  //}
}

static void printVector(const vector<uint>& m) {
  vector<uint>::size_type size = m.size();
  cout << "[";
  if (size) {
    ostream_iterator<uint> out_it (cout,", ");
    copy(m.begin(), m.begin() + size - 1, out_it);
    cout << m[size - 1];
  }
  cout << "]";
}

static void printVector(uint* m, const uint size) {
  cout << "[";
  if (size) {
    ostream_iterator<uint> out_it (cout,", ");
    copy(m, m + size - 1, out_it);
    cout << m[size - 1];
  }
  cout << "]";
}

void printMatrix(uint* m, const uint rows, const uint cols) {
  printf("[");
  for (uint i = 0; i < rows; i++) {
    if (i > 0) {
      printf(" ");
    }
    printVector(&m[i * cols], cols);
    if (i < rows - 1) {
      printf("\n");
    }
  }
  printf("]\n");
}

void checkGPUConfiguration() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    cerr << "There is no device supporting CUDA\n" << endl;
    exit(-1);
  }
  else{
  	cout<<"Present";
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    cerr << "There is no CUDA capable device" << endl;
    exit(-1);
  }
  if ((WARP_SIZE != 32)) {
    cerr << "Warp size must be 32" << endl ;
    exit(-1);
  }
  // Make printf buffer bigger, otherwise some printf messages are not displayed
  size_t limit;
  cudaThreadGetLimit(&limit, cudaLimitPrintfFifoSize); 
  cudaThreadSetLimit(cudaLimitPrintfFifoSize, limit * 16);
  // Make stack bigger, otherwise recursive functions will fail silently (?)
  //cudaThreadGetLimit(&limit, cudaLimitStackSize);
  //cudaThreadSetLimit(cudaLimitStackSize, limit * 8);
}

int main(int argc, char const *argv[])
{
	/* code */
	checkGPUConfiguration();

	return 0;
}