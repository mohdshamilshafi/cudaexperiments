#!/bin/bash


/usr/local/cuda/bin/nvcc graphMinMaxDecremental.cu

for i in `seq 1 2`;
	do
		./a.out < ../GraphsUndirectedDynamic/Newest/web-BerkStan.txt
	done


for i in `seq 1 10`;
	do
		./a.out < ../GraphsUndirectedDynamic/Newest/wiki-Talk_m.txt
	done
	
	
#for i in `seq 1 30`;
#        do
#                ./a.out <Graphs/New/gcol$i.txt >Graphs/Newest/gcol$i.txt
#        done
#./a.out <Graphs/New/amazon.txt >Graphs/Newest/amazon.txt
#./a.out <Graphs/New/facebook.txt >Graphs/Newest/facebook.txt
#./a.out <Graphs/New/googleGraph.txt >Graphs/Newest/googleGraph.txt        
      
