#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <map>

using namespace std;


int main() {

//	string s;
//	
//	for (int i=0; i<4; i++){
//		getline (cin, s);
//		//cout<<s<<endl;
//	}

	int n, m;
	
	cin>>n>>m;
	
	cout<<n<<"\t"<<m<<endl;

//	cout<<"4847571\t68993773"<<endl;
	
	int count = 0;
	
	for (int i=0; i<m; i++){
		int a, b;
		
		cin>>a>>b;
		
		if (a!=b){
			count++;
			cout<<a+1<<"\t"<<b+1<<endl;
		}
	}
	//cout<<count;
	

	
 	return 0;
}
