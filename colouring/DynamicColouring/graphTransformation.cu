#include <iostream>
#include <thrust/sort.h>

using namespace std;

bool searchFunction(int *array, int *array2, int k, int m){
	int first = array2[k];
	int second = array[k];
	
	for (int i=0; i<m; i++){
		if (array[i]>first){
			return false;
		}
		
		else if (array[i]==first){
			if (array2[i]==second){
				return true;
			}
		}
	}
	
	return false;
}

int main(int argc, char const *argv[])
{
	/* code */
	string a, b;
	int n, m;
	cin>>a>>b>>n>>m;
	
	int *array = new int [m];
	int *array2 = new int [m];
	
	
	cout<<a<<" "<<b<<" "<<n<<" "<<m<<endl;

	for (int i = 0; i < m; ++i)
	{
		/* code */
		int start;
		int end;
//		char c;

		cin>>start>>end;
		
		array[i]=start;
		array2[i]=end;
//		
	}	
	
//	for (int i=0; i<2*m; i++){
//		cout<<array[i]<<"\t"<<array2[i]<<endl;
//	}
	
	thrust::sort_by_key(array2, array2 + m, array);
	thrust::sort_by_key(array, array + m, array2);
	
	int count = 1;
	
//	cout<<array[0]<<"\t"<<array2[0]<<endl;
	
	for (int i=1; i<m; i++){
		if (array[i] < array2[i]){
//			cout<<array[i]<<"\t"<<array2[i]<<endl;
			count++;
		}
		
		else if (!searchFunction(array, array2, i, m)){
//			cout<<array[i]<<"\t"<<array2[i]<<endl;
			count++;
		}
	}
	
	cout<<"m "<<count<<endl;
	
	return 0;
}
