#!/bin/bash
for i in `seq 20 30`;
        do
                ./a.out <GraphsUndirected/gcol$i.txt >GraphsUndirected/Output/gcol$i.txt
        done
./a.out <GraphsUndirected/facebook.txt >GraphsUndirected/Output/facebook.txt
./a.out <GraphsUndirected/sample.txt >GraphsUndirected/Output/sample.txt
./a.out <GraphsUndirected/amazon.txt >GraphsUndirected/Output/amazon.txt        
      
