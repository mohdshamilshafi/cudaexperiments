#include <iostream>

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

void edgesPrintNew (int vertexArray[], int neighbourArray[], int n, int m){ 

//	for (int i=0; i<n; i++){
//		cout<<vertexArray[i]<<" ";
//	}
//	
//	cout<<endl;


	for (int i=0; i<m; i++){
		cout<<neighbourArray[i]<<" ";
	}
	
	cout<<endl;
}

int main(int argc, char const *argv[])
{
	/* code */
	string a, b;
	int n, m;
	
	cin>>a>>b>>n>>m;
	
	cout<<a<<" "<<b<<" "<<n<<" "<<m<<endl;

	int vertexArray[n];
	int neighbourArray[m];

	for (int i = 0; i < n; ++i)
	{
		/* code */
		vertexArray[i]=m;
	}

	int offset = 0;

	int current = 0;
	int mark = 1;

	for (int i = 0; i < m; ++i)
	{
		/* code */
		
		int start;
		int end;

		cin>>start>>end;
//		
//		cin>>start>>end;
//		start++;
//		end++;

		if (start!=mark){ 

			if (start == mark+1 && vertexArray[mark-1]!=m){ 

			}

			else{

				for (int j = mark; j<start; j++){ 
					vertexArray[j-1]=offset;
					// neighbourArray[offset]=0;
					// offset++;
				}
			}
			mark = start;

		}

		if (start==current){ 
			neighbourArray[offset]=end;
			offset++;
		}

		else { 
			current = start;

			vertexArray[current-1]=offset;

			neighbourArray[offset]=end;
			offset++;
		}
	}

	edgesPrintNew(vertexArray, neighbourArray, n, m);

	
	int graphDegree[n+1];
	
	for (unsigned int i = 0; i < n+1; i += 1)
	{
		graphDegree[i]=0;
	}
	
	for (int i=0; i<n-1; i++){
		for (int j = vertexArray[i]; j < vertexArray[i+1]; ++j){
				graphDegree[i]++;
				graphDegree[neighbourArray[j]-1]++;
				/* code */
		}
	}

	for (int j = vertexArray[n-1]; j < m; ++j)
	{
			graphDegree[n-1]++;
			graphDegree[neighbourArray[j]-1]++;
				/* code */
	}	
	
	for (unsigned int i = 0; i < n; i += 1)
	{
		if (graphDegree[n]<graphDegree[i]){
			graphDegree[n]=graphDegree[i];
		}
	}
	
	
//	for (unsigned int i = 0; i < n+1; i += 1)
//	{
//		cout<<graphDegree[i]<<endl;
//	}
	
	cout<<graphDegree[n];
	
	return 0;
}
