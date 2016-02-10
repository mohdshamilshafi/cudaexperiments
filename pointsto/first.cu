#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

using namespace std;
#define B2MB(x) ((x)/(1024 * 1024))
    
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

int main(int argc, char const *argv[])
{
	/* code */
	printDeviceMemory();
	return 0;
}