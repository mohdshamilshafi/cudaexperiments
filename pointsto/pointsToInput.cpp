#include <iostream>
#include <map>

using namespace std;

void edgesPrint (int vertexArray[], int neighbourArray[], int n, int m){ 

	for (int i=0; i<n-1; i++){
		for (int j = vertexArray[i]; j < vertexArray[i+1]; ++j){

				cout<<"e "<<i+1<<" "<<neighbourArray[j]<<endl;
				/* code */
		}
	}

	for (int j = vertexArray[n-1]; j < m; ++j)
	{
			cout<<"e "<<n<<" "<<neighbourArray[j]<<endl;
				/* code */
		
	}
}

int main(int argc, char const *argv[])
{
	int n, m;
	
	cin>>m;
	
	int nodeCount = 0;
	
	map<string, int> variablesMap;
	map<string,int>::iterator it;
	
	for (int i=0; i<m; i++){
		string a, b, temp;
		char c;
		
		cin>>a>>c>>b;
		
		// Address
		if (b[0]=='&'){
			temp = b.substr(1);
			
			it = variablesMap.find(a);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(a,nodeCount) );
				nodeCount++;
			}
			
			it = variablesMap.find(temp);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(temp,nodeCount) );
				nodeCount++;
			}
			
		}
		
		// Load
		else if (b[0]=='*'){
			temp = b.substr(1);
			
			
			it = variablesMap.find(a);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(a,nodeCount) );
				nodeCount++;
			}
			
			it = variablesMap.find(temp);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(temp,nodeCount) );
				nodeCount++;
			}
		}
		
		// Store
		else if (a[0]=='*'){
			temp = a.substr(1);
			
			
			it = variablesMap.find(b);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(b,nodeCount) );
				nodeCount++;
			}
			
			it = variablesMap.find(temp);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(temp,nodeCount) );
				nodeCount++;
			}
		}
		
		// Copy
		else{
		
			
			it = variablesMap.find(a);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(a,nodeCount) );
				nodeCount++;
			}
			
			it = variablesMap.find(b);
			if (it == variablesMap.end()){
				variablesMap.insert( pair<string,int>(b,nodeCount) );
				nodeCount++;
			}
		
		}	
	}
	
	n = nodeCount;
	
	for (it=variablesMap.begin(); it!=variablesMap.end(); ++it)
    		cout << it->first << " => " << it->second << '\n';
	
	return 0;
}
