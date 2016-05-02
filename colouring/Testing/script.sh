#!/bin/bash

/usr/local/cuda/bin/nvcc graphRandomColouringUndirected.cu

for i in ../GraphsUndirected/New/*
do
	filename=$(basename "$i")
	filename="${filename%.*}"
	echo "$filename;"
	for i in `seq 1 5`;
		do
			./a.out < ../GraphsUndirected/New/$filename.txt
		done
	#wget ftp://ftp.cmbi.ru.nl//pub/molbio/data/dssp/$filename.dssp
	#echo ${i##*/}
done

/usr/local/cuda/bin/nvcc graphNewMaxColouringRandom.cu

for i in ../GraphsUndirected/New/*
do
	filename=$(basename "$i")
	filename="${filename%.*}"
	echo "$filename;"
	for i in `seq 1 5`;
		do
			./a.out < ../GraphsUndirected/New/$filename.txt
		done
	#wget ftp://ftp.cmbi.ru.nl//pub/molbio/data/dssp/$filename.dssp
	#echo ${i##*/}
done


/usr/local/cuda/bin/nvcc graphNewMinMaxColouringRandom.cu

for i in ../GraphsUndirected/New/*
do
	filename=$(basename "$i")
	filename="${filename%.*}"
	echo "$filename;"
	for i in `seq 1 5`;
		do
			./a.out < ../GraphsUndirected/New/$filename.txt
		done
	#wget ftp://ftp.cmbi.ru.nl//pub/molbio/data/dssp/$filename.dssp
	#echo ${i##*/}
done

#/usr/local/cuda/bin/nvcc graphMinMaxIncrementalNew.cu

#for i in ../GraphsUndirectedDynamic/Newest/*
#do
#	filename=$(basename "$i")
#	filename="${filename%.*}"
#	echo "$filename;"
#	for i in `seq 1 5`;
#		do
#			./a.out < ../GraphsUndirectedDynamic/Newest/$filename.txt
#		done
#	#wget ftp://ftp.cmbi.ru.nl//pub/molbio/data/dssp/$filename.dssp
#	#echo ${i##*/}
#done

#/usr/local/cuda/bin/nvcc graphMinMaxDecrementalNew.cu

#for i in ../GraphsUndirectedDynamic/Newest/*
#do
#	filename=$(basename "$i")
#	filename="${filename%.*}"
#	echo "$filename;"
#	for i in `seq 1 5`;
#		do
#			./a.out < ../GraphsUndirectedDynamic/Newest/$filename.txt
#		done
#	#wget ftp://ftp.cmbi.ru.nl//pub/molbio/data/dssp/$filename.dssp
#	#echo ${i##*/}
#done
#for i in `seq 1 30`;
#        do
#                ./a.out <Graphs/New/gcol$i.txt >Graphs/Newest/gcol$i.txt
#        done
#./a.out <Graphs/New/amazon.txt >Graphs/Newest/amazon.txt
#./a.out <Graphs/New/facebook.txt >Graphs/Newest/facebook.txt
#./a.out <Graphs/New/googleGraph.txt >Graphs/Newest/googleGraph.txt        
      
