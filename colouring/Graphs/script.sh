#!/bin/bash

for i in `seq 1 30`;
        do
                ./a.out <gcol$i.txt >New/gcol$i.txt
        done
        
#./a.out <../Graphs/amazon.txt >../GraphsUndirectedDynamic/amazon.txt
#./a.out <../Graphs/facebook.txt >../GraphsUndirectedDynamic/facebook.txt
