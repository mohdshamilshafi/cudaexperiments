#!/bin/bash

for i in ../GraphsUndirectedDynamic/Newest/*
do
	filename=$(basename "$i")
	filename="${filename%.*}"
	echo "$filename;"
	./a.out < ../GraphsUndirectedDynamic/Newest/$filename.txt
	#wget ftp://ftp.cmbi.ru.nl//pub/molbio/data/dssp/$filename.dssp
	#echo ${i##*/}
done

