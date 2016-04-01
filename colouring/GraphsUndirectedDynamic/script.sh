#!/bin/bash

for i in `seq 1 30`;
        do
                diff Old/gcol$i.txt New/gcol$i.txt > gcol$i.txt
        done
       
diff Old/amazon.txt New/amazon.txt > amazon.txt
diff Old/facebook.txt New/facebook.txt > facebook.txt
