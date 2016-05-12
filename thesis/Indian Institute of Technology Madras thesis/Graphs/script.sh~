#!/bin/bash


#pdfcrop chart16.pdf chart16New.pdf
#pdftk chart16New.pdf cat 1west output chart16Newest.pdf 


for i in `seq 1 37`;
        do
#        		pdfcrop chart$i.pdf chartNew$i.pdf
#				pdftk chartNew$i.pdf cat 1west output chartNewest$i.pdf 
        		convert -density 150 chartNewest$i.pdf -quality 90 chartNewest$i.jpg

        done
