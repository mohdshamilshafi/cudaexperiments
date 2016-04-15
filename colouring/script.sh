#!/bin/bash

for i in Graphs/New/*
do
	filename=$(basename "$i")
	filename="${filename%.*}"
	#echo "$filename"
	./a.out < Graphs/New/$filename.txt > GraphsUndirected/New/$filename.txt
	#wget ftp://ftp.cmbi.ru.nl//pub/molbio/data/dssp/$filename.dssp
	#echo ${i##*/}
done

#for i in `seq 1 30`;
#        do
#                ./a.out <Graphs/New/gcol$i.txt >Graphs/Newest/gcol$i.txt
#        done
#./a.out <Graphs/New/amazon.txt >Graphs/Newest/amazon.txt
#./a.out <Graphs/New/facebook.txt >Graphs/Newest/facebook.txt
#./a.out <Graphs/New/googleGraph.txt >Graphs/Newest/googleGraph.txt        
      
