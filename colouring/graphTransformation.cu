#include <iostream>
#include <thrust/sort.h>
#include <set>

using namespace std;

int main(int argc, char const *argv[])
{
	/* code */
	string a, b;
	int n, m;
	cin>>a>>b>>n>>m;
//	cin>>n>>m;
	
	int *array = new int [m];
	int *array2 = new int [m];
	
	
	cout<<n<<" "<<m<<endl;

	for (int i = 0; i < m; ++i)
	{
		/* code */
		int start;
		int end;
		char c;
		
		

		cin>>c>>start>>end;
		
		
		array[i]=start;
		array2[i]=end;
		
//		array[2*i+1]=end;
//		array2[2*i+1]=start;
		
//		
	}	
	
//	for (int i=0; i<2*m; i++){
//		cout<<array[i]<<"\t"<<array2[i]<<endl;
//	}
	
	thrust::sort_by_key(array2, array2 + m, array);
	thrust::sort_by_key(array, array + m, array2);
	
//	int prev1 = array[0];
//	int prev2 = array2[0];
//	
//	
//	
//	cout<<array[0]<<"\t"<<array2[0]<<endl;
//	
//	int count = 1;
//	
//	for (int i=1; i<m; i++){
//		if (prev1 == array[i] && prev2 == array2[i]){
//			continue;
//		}
//		cout<<array[i]<<"\t"<<array2[i]<<endl;
//		count++;
//		prev1 = array[i];
//		prev2 = array2[i];
//	}
	
	typedef pair<int, int> pairs;
	
	pairs temp;
	
	set<pairs> setTemp;
	set<pairs> :: iterator it;
	
	int count = 0;
	
	for (int i=0; i<m; i++){
		if (array[i]<array2[i]){
			temp.first = array[i];
			temp.second = array2[i];
		}
		else{
			temp.first = array2[i];
			temp.second = array[i];
		}
		
		it = setTemp.find(temp);
		
		if (it==setTemp.end()){
			setTemp.insert(temp);
			cout<<array[i]<<"\t"<<array2[i]<<endl;
			count++;
		}
		
	}
	
	//cout<<count<<endl;
	
	return 0;
}
