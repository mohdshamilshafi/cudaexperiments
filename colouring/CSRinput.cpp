#include <iostream>

int main(int argc, char const *argv[])
{
	/* code */
	int n, m;
	
	cin>>n>>m;

	int vertexArray[n];
	int neighbourArray[m];

	for (int i = 0; i < n; ++i)
	{
		/* code */
		vertexArray[i]=m;
	}

	int offset = 0;

	int current = 0;

	for (int i = 0; i < m; ++i)
	{
		/* code */
		char c;
		int start;
		int end;

		cin>>c>>start>>end;

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

	return 0;
}