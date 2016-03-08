#include <iostream>
#include <thrust/sort.h>

using namespace std;

int main(int argc, char const *argv[])
{
	/* code */
	string a, b;
	int n, m;
	cin>>a>>b>>n>>m;
	
	int *array = new int [2*m];
	int *array2 = new int [2*m];
	
	
	cout<<a<<" "<<b<<" "<<n<<" "<<m<<endl;

	for (int i = 0; i < m; ++i)
	{
		/* code */
		int start;
		int end;

		cin>>start>>end;
		
		array[2*i]=start;
		array2[2*i]=end;
		
		array[2*i+1]=end;
		array2[2*i+1]=start;
		
//		
	}	
	
//	for (int i=0; i<2*m; i++){
//		cout<<array[i]<<"\t"<<array2[i]<<endl;
//	}
	
	thrust::sort_by_key(array2, array2 + 2*m, array);
	thrust::sort_by_key(array, array + 2*m, array2);
	
	int prev1 = array[0];
	int prev2 = array2[0];
	
	cout<<array[0]<<"\t"<<array2[0]<<endl;
	
	for (int i=1; i<2*m; i++){
		if (prev1 == array[i] && prev2 == array2[i]){
			continue;
		}
		cout<<array[i]<<"\t"<<array2[i]<<endl;
		prev1 = array[i];
		prev2 = array2[i];
	}
	
	return 0;
}
