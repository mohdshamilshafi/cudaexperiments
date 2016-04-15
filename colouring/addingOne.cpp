#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <map>

using namespace std;


int main() {

	map <int, int> newMap;
	map <int, int>::iterator it;
	int count = 1;	
	
	int n, m;
	cin>>n>>m;
	
 	int a, b;
 	
// 	while (cin>>a>>b){
// 		it = newMap.find(a);
// 		
// 		if (it == newMap.end()){
// 			newMap.insert(pair<int,int>(a, count));
// 			cout<<count<<"\t";
// 			count++;
// 		}
// 		
// 		else{
// 			cout<<it->second<<"\t";
// 		}
// 		
// 		it = newMap.find(b);
// 		
// 		if (it == newMap.end()){
// 			newMap.insert(pair<int,int>(b, count));
// 			cout<<count<<endl;
// 			count++;
// 		}
// 		
// 		else{
// 			cout<<it->second<<endl;
// 		}
// 	}
 	
 	
 	int min = 2000000000;
 	int max = 0;
 	
 	bool first = true;
 	
 	while (cin>>a){
 		
// 		if (first){
// 			cout<<a-11<<"\t";
// 			first = false;
// 		}
// 		
// 		else{
// 			cout<<a-11<<endl;
// 			first = true;
// 		}
 		
 		if (min > a){
 			min = a;
 		}
 		
 		if (max < a){
 			max = a;
 		}
 	}
 	
 	cout<<min<<" "<<max;

//	cout<<count; 	
 	return 0;
}
