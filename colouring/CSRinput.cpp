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
	
	int NSlast = 0;
	int NSoffset = 0;
	int NSprev=0;
	
	
	for (int i=0; i<m; i++){
		int start, end;
		cin>>start>>end;
		
		for (int j=NSlast+1; j<start; j++){
			vertexArray[j-1]=NSoffset;
			
		}
		
		if (NSprev!=start){
			NSlast=start;
			vertexArray[start-1]=NSoffset;
			NSprev=start;
		}
		
		neighbourArray[NSoffset]=end;
		NSoffset++;
		
		
	}

//	int offset = 0;

//	int current = 0;
//	int mark = 1;

//	for (int i = 0; i < m; ++i)
//	{
//		/* code */
//		char c;
//		int start;
//		int end;

//		cin>>c>>start>>end;

//		if (start!=mark){ 

//			if (start == mark+1 && vertexArray[mark-1]!=m){ 

//			}

//			else{

//				for (int j = mark; j<start; j++){ 
//					vertexArray[j-1]=offset;
//					// neighbourArray[offset]=0;
//					// offset++;
//				}
//			}
//			mark = start;

//		}

//		if (start==current){ 
//			neighbourArray[offset]=end;
//			offset++;
//		}

//		else { 
//			current = start;

//			vertexArray[current-1]=offset;

//			neighbourArray[offset]=end;
//			offset++;
//		}
//	}

	for (int i = 0; i < n; ++i)
	{
		cout<<vertexArray[i]<<" ";
		/* code */
	}

	cout<<endl;

	for (int i = 0; i < m; ++i)
	{
		cout<<neighbourArray[i]<<" ";
		/* code */
	}

	cout<<endl;

	edgesPrint(vertexArray, neighbourArray, n, m);



	return 0;
}
