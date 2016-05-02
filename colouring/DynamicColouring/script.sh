#!/bin/bash

g++ graphTransformationDegree.cpp

for i in ../Graphs/New/*
do
	filename=$(basename "$i")
	filename="${filename%.*}"
	#echo "$filename"
#	./a.out <../Graphs/New/$filename.txt >../GraphsUndirectedDynamic/gcol$i.txt
	./a.out < ../Graphs/New/$filename.txt > ../GraphsUndirectedDynamic/Newest/$filename.txt
	#wget ftp://ftp.cmbi.ru.nl//pub/molbio/data/dssp/$filename.dssp
	#echo ${i##*/}
done



#for i in `seq 1 30`;
#        do
#                ./a.out <../Graphs/gcol$i.txt >../GraphsUndirectedDynamic/gcol$i.txt
#        done
        
#./a.out <../Graphs/amazon.txt >../GraphsUndirectedDynamic/amazon.txt
#./a.out <../Graphs/facebook.txt >../GraphsUndirectedDynamic/facebook.txt
