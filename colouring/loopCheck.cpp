#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <map>

using namespace std;


int main() {
	int n, m;
	
	cin>>n>>m;
	int a, b;
	
 	while (cin>>a>>b){
 		
// 		if (first){
// 			cout<<a-11<<"\t";
// 			first = false;
// 		}
// 		
// 		else{
// 			cout<<a-11<<endl;
// 			first = true;
// 		}
 		if (a==b){
 			cout<<a<<" "<<b<<endl;
 			cout<<"Loop found. Exiting.";
 			return 0;
 		}
 	}
 	
 	
 	return 0;
}
