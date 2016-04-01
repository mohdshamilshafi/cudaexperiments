#!/bin/bash

g++ graphTransformationDegree.cpp

for i in `seq 1 30`;
        do
                ./a.out <../Graphs/gcol$i.txt >../GraphsUndirectedDynamic/gcol$i.txt
        done
        
#./a.out <../Graphs/amazon.txt >../GraphsUndirectedDynamic/amazon.txt
#./a.out <../Graphs/facebook.txt >../GraphsUndirectedDynamic/facebook.txt
