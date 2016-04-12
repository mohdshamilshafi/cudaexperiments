#!/bin/bash

for i in `seq 1 10`;
        do
                ./a.out <output >outputcheck$i
        done
        
#./a.out <../Graphs/amazon.txt >../GraphsUndirectedDynamic/amazon.txt
#./a.out <../Graphs/facebook.txt >../GraphsUndirectedDynamic/facebook.txt
